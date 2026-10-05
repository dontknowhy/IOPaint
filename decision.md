# 最终决定：IOPaint 审计修复方案

> 本文取代此前的审计草稿，是基于你 8 条回复后的**最终实施方案**。
> 原则：只读阶段结束，进入修复阶段；每项给出 改动点(file:line) → 方案 → 验收测试。
> 归属标注沿用审计结论：除 PERF-1/API-1/PKG-1 外均为上游继承（上游 archived，不再考虑向上游提 PR，见 D-16）。

---

## 0. 你的回复 → 我的决定

| # | 你的回复 | 决定 |
|---|---|---|
| 1/2 | 只暴露局域网，但安全不能妥协，得修 | **SEC-1/SEC-2 路径遍历、DOS-1 输入边界全修（P0）**。即使暂未启用 FileManager `--input`，也一并修：成本低、是上游继承的洞，启用即暴露 |
| 3 | quality 默认应为 95，保持可配置 | `cli.py:129` `Option(100 → 95)`；`const.py:123`、`helper.py:147`、`web_config.py:49` 已是 95，改完三处一致 |
| 4 | 倾向响应速度；问"内部不会真的 PNG 到处传吧？" | **是的，内部不传 PNG**（见 §1）。决定：PNG `compress_level=1`、JPEG 默认 95、编码全部移出事件循环 |
| 5 | 删除 | **删除 `sd_mask_blur` 功能**（schema + 模型代码 + 前端 UI + payload），不是只删注释。见 D-17，含行为影响说明 |
| 6 | 批处理同名冲突加后缀并 warning | `glob_images` 同 stem 加 `_1/_2` 后缀 + `logger.warning`；mask 查找回退基础 stem；输出覆盖原图时同样加后缀。见 D-15 |
| 7 | 可以 | 接受：① 钉死 `opencv-python`、`transformers` 版本；② CI 增加**非阻塞**的慢速冒烟（nightly + 手动触发，不进 PR 必跑集）。见 D-16 |
| 8 | 上游 archive 无人维护 | 不提上游 PR；修复全部落在本 fork；"上游继承/自引入"标注仅用于排查参考，不作为约束 |

---

## 1. 回答你的疑问：内部会不会到处传 PNG？（#4）

**不会。** 全部内部管线是 numpy / torch tensor，PNG 只出现在 4 个边界：

1. **请求体**：前端 `canvasToBlob` → base64（`api.ts:60`），服务端 `decode_base64_to_image` 解一次码；
2. **响应体**：`pil_to_bytes` / `numpy_to_bytes` 编一次码（`api.py:367-378`、`api.py:404,429`、`api.py:440`）；
3. **mask 往返**：前端 mask PNG blob ↔ `adjust_mask` / `gen_mask` 的 PNG 响应（`api.py:430,440`）；
4. **落盘**：批处理输出（`batch_processing.py:117`）、缩略图（`file_manager/utils.py:14`）。

模型内部 `forward()` 之间、HD strategy 各阶段、调度器之间全部是数组。所以**调 PNG 压缩级别只影响线上字节量和编码耗时，不影响任何内部精度**——PNG 本身无损，放心降级。

---

## 2. 具体方案

### Phase 1（P0 安全 + 速赢）

**D-1 路径遍历（SEC-1/SEC-2）**
- `iopaint/file_manager/file_manager.py:74-78` `_get_file`：`p = (base / filename).resolve()`，再 `p.is_relative_to(base.resolve())` 否则 `422`；`base is None`（未配置 mask/output dir）时同样 422 而不是 TypeError。
- `file_manager.py:119-177` `get_thumbnail`：读取 `original_filepath` 前（`:130`）对 `filename` 做同样的 resolve + is_relative_to 校验——thumbnail 文件名本身是 md5（安全），危险的是拼出来的 `original_path`。
- 测试 `iopaint/tests/test_file_manager_security.py`：tmp 目录下 `filename=../../etc/passwd`、绝对路径 `/etc/passwd`、`tab=mask` 未配置，全部断言 4xx 且文件未被读取。

**D-2 输入边界（DOS-1）** — `iopaint/schema.py`
- `sd_steps(:344)` → `Field(50, ge=1, le=150)`；`ldm_steps(:290)` → `Field(20, ge=1, le=150)`（前端 slider max=100，留余量）。
- `kernel_size(:512)` → `ge=1, le=1000`；`croper_x/y/w/h(:316-319)`、`extender_x/y/w/h(:324-327)` → `ge=-16384, le=16384`（extender 坐标可为负，见 `base.py:296` 注释），handler 内再对照 `image.shape` 校验。
- `image/mask(:287-288)` → `max_length=64_000_000`（≈48MB 二进制，覆盖 12MP PNG b64 ≈15MB）；`api_inpaint` 对 `None` 显式 400 而非 500。
- `iopaint/__init__.py`：`Image.MAX_IMAGE_PIXELS = None` 改为显式上限（建议 100MP），防解压炸弹。
- `api.py` 加中间件：`Content-Length > 256MB → 413`。
- `clicks(:455)` 已是 `List[List[int]]`；pydantic v2 lax 模式对小数 int 的行为**待验证**（见 §4），无论结果如何前端都先 round（D-11）。

**D-3 CORS 最小加固（SEC-3）** — `api.py:161-168`
- `allow_credentials=False`（当前 `origins=["*"] + credentials=True` 是规范无效组合，且全站无 cookie/鉴权，无损失）；origins 保持 `*`（局域网多来源访问 + dev 跨源 5173 需要）。真正防线是 D-1/D-2。

**D-4 事件循环阻塞（PERF-1，fork 引入的回归）** — `api.py`
- `api_inpaint(:343)`：把 decode(:344-353)、post-process+encode(:363-376)、`torch_gc` 全部包进一个同步函数 `_inpaint_blocking(req)`，一次 `run_in_executor` 跑完；`sio.emit("diffusion_finish")` 留在 async 侧。`HTTPException` 在 executor 内抛出会正常穿透 await。
- `api_run_plugin_gen_image(:382)`、`api_run_plugin_gen_mask(:414)` 同样处理（decode :390/:421、encode :404/:429 目前都在 loop 上）。
- `api_adjust_mask(:437)` 是 `def`（FastAPI 走线程池），不用动。
- 验收：并发 4 个 GET 请求打 inpaint 大图，期间 `/api/v1/samplers` 响应 p95 应 <100ms（当前实测 max 4696ms）。

**D-5 编码参数（PERF-2 + quality）**
- `helper.py:132-139 numpy_to_bytes`：按 ext 传参——JPEG `[IMWRITE_JPEG_QUALITY, quality]`、PNG `[IMWRITE_PNG_COMPRESSION, 6]`（mask 是平坦图，实测 comp=0 → 12.02MB、comp=6 → 0.01MB，编码成本可忽略）；删除现在 JPEG/PNG 参数混传的写法。
- `helper.py:147 pil_to_bytes`：新增 `png_compress_level` 参数，**API 响应默认 1**（实测 12MP：level1=1158ms / level6=7247ms / level9=36964ms，体积仅 +18%）；批处理可后续按需调回 6。
- `cli.py:129` `quality: Option(100 → 95)`。
- 验收：`test_helper.py` 补断言 JPEG 字节 q95、PNG level1 解码后像素与原图逐像素相等（无损）。

**D-6 扩展名/Content-Type 一致性（CONS-3）**
- `file_manager.py:44`：`media_type` 用 `mimetypes.guess_type` 按真实扩展名给（jpeg→image/jpeg），或读魔数兜底。
- `file_manager.py:47-62` + `file_manager/utils.py:14`：缩略图**文件名后缀跟内容一致**——`Image.open` 后 `format` 即可得（不需 load），把 `md5 + ".jpg"` 改成 `md5 + "." + 真实 ext`；旧缩略图缓存自然失效重建。`media_type` 同步按后缀给。
- 前端保存：`Editor.tsx:1038-1067` 与 `api.ts:190-199` 的保存文件名，扩展名以**响应的 Content-Type / blob.type** 为准（服务端响应格式跟随输入字节 `helper.py:327`，而 `file.name` 可能是被改过名的伪扩展名——这是实测复现的错配根源）。
- 前端 mask 下载（`Editor.tsx:1043,1051`）由 `.jpg`/`image/jpeg` 改为 `.png`/`image/png`：mask 是 0/255 二值图，JPEG 边缘污染不可接受。
- 批处理输出（`batch_processing.py:117-118`）恒 PNG 且名实一致，不动。

**D-7 SystemExit 杀进程（REL-1）**
- `helper.py:68`、`helper.py:97`、`model/utils.py:1003`、`plugins/base_plugin.py:16`：改为自定义 `ModelLoadError(RuntimeError)`；`api.py` 的 exception middleware 捕获 → 500 + 中文提示；`cli.py` 启动路径 catch 后 `logger.error + SystemExit(1)`（启动期语义不变）。
- `cli.py:161-169`、`batch_processing.py:50` 是纯 CLI 路径，保留 SystemExit。

**D-8 视图写穿 / 直方图除零（速赢）**
- `base.py:104`、`base.py:286`：`inpaint_result = image[:, :, ::-1]` 是负步长**视图**，随后 `inpaint_result[...] = ...` 会写穿调用方的 `image` → 加 `.copy()`。
- `base.py:195-198,215-221`：`mask==0` 区域为空时 `cdf[-1]=0` → 除零 → NaN → 全黑图。`_calculate_cdf` 对零和加保护，或 `_match_histograms` 对空选区直接返回原图。
- `file_manager.py:162`：`self.app.logger` 不存在（AttributeError 会吞掉原始错误）→ 改 loguru `logger.warning`。

### Phase 2（可靠性）

**D-9 模型切换状态机（REL-2）** — `model_manager.py`
- `api_switch_model` 先校验 `name in available_models` → 422（现在 KeyError → 500，且 `:135` 已把 `self.name` 改掉）。
- `switch(:128-158)`：`self.name` / `self.controlnet_method` 的提交移到 `init_model` **成功之后**；`except` 分支保留重建旧模型的回滚，若回滚也失败则 `self.model = None`。
- `_run(:110)` 开头加 `if self.model is None: raise HTTPException(503, "model load failed, please switch model")`，避免 `del self.model` 后的 AttributeError 级联。
- `switch_controlnet_method(:219-236)` 同模式：先算局部变量，成功后提交。
- 测试：monkeypatch `init_model` 抛异常，断言 `self.name` 不变、后续调用拿到 503/4xx 而非崩溃。

**D-10 插件并发锁（CON-1）** — `plugins/base_plugin.py` + `api.py`
- `BasePlugin` 加 `self.lock = threading.Lock()`；`api_run_plugin_gen_image/gen_mask/switch_plugin_model` 三个入口在 executor 内 `with plugin.lock:` 包住调用（与 `ModelManager.lock` 同一模式，锁在组件内/调用点，勿在 async 层持锁）。
- 顺手把 `interactive_seg.py:136-144` 的 `prev_img_md5` 赋值挪到 `set_image` 成功之后（check-then-act 顺序修正）。

**D-11 前端状态卡死（FE-1）**
- `states.ts` `adjustMask(:1088)`：`set(isAdjustingMask=true)` 后整段包 `try/finally → set(false)`，覆盖 `:1115` 文件切换早退。
- `states.ts` `runInpainting(:473)`：同上；`:532`、`:554` 两处早退目前绕过了 `:558` 的复位，`finally` 统一兜底（`isInpainting` 是全局标志，必须无条件复位；`temporaryMasks` 仍按 `file === file` 判断后清）。
- `states.ts` `runRenderablePlugin(:564)`、`Editor.tsx:752 runInteractiveSeg`：`getCurrentTargetFile()`（Editor.tsx:754）移进 try，`isPluginRunning` 放 finally。

**D-12 点击坐标取整（API-1）** — 在 `runInteractiveSeg` 入口统一 `newClicks.map(([x,y,...r]) => [Math.round(x), Math.round(y), ...r])`，一次覆盖 `Editor.tsx:825/827/1593` 三个来源（含触屏），不动笔迹绘制的浮点精度。

**D-13 422 错误展示（CONS-2）** — `api.ts:28-49` 拦截器：`detail` 为数组时（FastAPI 校验错误形状 `[{loc,msg,type}]`）格式化为 `loc: msg; ...`，否则维持现状，杜绝 `[object Object]`。

**D-14 空列表/空渲染守卫（FE-2）**
- `Editor.tsx:1013-1040 download`：`renders.length === 0` 时跳过（或回落下载原图），修 Ctrl+S 崩溃。
- `FileManager` 方向键：`filenames.length === 0` 时忽略。
- `api.py:258-275 save_image`：把 `:266` 的 output_dir 存在性/None 校验挪到 `:263` 构造路径之前（现在 None 时 `Path / str` 先抛 TypeError）。

**D-15 批处理同名（你的 #6）** — `batch_processing.py`
- `glob_images(:25-34)`：stem 已存在时 key 改为 `{stem}_{n}` + `logger.warning("duplicate stem ...")`。
- mask 查找（`:74-95` 附近）：先精确匹配，miss 后回退 `stem.rsplit("_", 1)[0]`（保证 `a.jpg + a.png` 共用 `a.png` mask 的配对不被后缀破坏）。
- 输出 `:117-120`：`save_p` 已存在且属于输入集合时 → 写 `{stem}_1.png` + warning（防 `--output-dir == --input-dir` 覆盖原图）。

**D-16 打包与 CI（你的 #7）**
- `setup.py:30-37`：`re.sub(r'(==[^;]+)\+[^;\s]+', r'\1', line)` 剥离 `+cu121` 本地版本号 → wheel 可从 PyPI 正常安装（PKG-1）。
- `requirements.txt:8,13`：钉死 `opencv-python==5.0.0.93`、`transformers==4.57.6`（与当前生产环境一致；`peft==0.20.0`、`diffusers==0.34.0` 已钉）。AGENTS 的"不许动 torch pin"不涉及这两行。
- CI 新增 `slow-smoke` job：`schedule(weekly)` + `workflow_dispatch`，`pytest -m slow`，HF/模型缓存用 actions/cache——**不进 PR 必跑集**，`check.sh` 与 PR CI 保持现状（快、离线）。

**D-17 删除 `sd_mask_blur`（你的 #5"删除"）**
- 删 `schema.py:335` 字段（pydantic v2 默认忽略未知字段，旧客户端 payload 兼容）。
- 删 `base.py:391-396` `forward_pre_process` 覆写（删后与基类默认实现一致）和 `base.py:402-405` 的 extender 模糊块。
- 删前端：`DiffusionOptions.tsx:765-790` 滑块、`states.ts:86,335`、`api.ts:91`。
- 删 `tests/test_brushnet.py:96`。
- **行为影响（已知并接受）**：BUG-3 证明该参数从未作用于模型输入（`base.py:68` 的 pre_process 结果没进 `:70` 的 forward），所以删除**不改变模型输出**；唯一变化是 `use_extender + sd_match_histograms` 时直方图匹配用的 mask 由"模糊后"变"原始"（`base.py:398-400,75-77`），边缘统计略有差异。
- 备选（若你本意是"搁置不修"）：只删死代码路径、保留字段为 no-op——告诉我即可，改动范围缩小到 schema+base.py。

**D-18 顺手修（P2，低成本）**
- `base.py:75-77`：`mask/255` 的 float64 → 显式 `astype(np.float32)`，12MP 图省 ~100MB 临时内存（BUG-5）。
- `ldm.py:311,320,324`：删除无条件 `torch.cuda.empty_cache()`（每推理两次全清，实测明显拖慢），需要时走 `--empty-cache-after-inpaint`（BUG-4）。
- `api.py:274` 等：`download.py:247,300` 模型扫描对单个坏目录 try/except 跳过并 warning，不因一个脏文件起不来（BUG-7）。

### Phase 3（一致性收尾）

**D-19 FileManager 竞态（FE-3）**：列表拉取加 AbortController / 请求序号，丢弃过期响应；`showPrevMask` 同样加序号守卫。放最后，纯健壮性。

---

## 3. 明确不做

- **不向上游提 PR**（archive，你的 #8）。
- **不加鉴权体系**：部署面仅局域网（你的 #1），修复以遍历 + 边界 + CORS 加固为界；如日后暴露公网需另立方案（Basic Auth/反代），本次不扩散范围。
- **不改 JSON/multipart 传输协议**：请求体 17MB 在 localhost 实测仅 ~110ms，瓶颈在编码不在传输。
- **不动 torch/torchvision 的 `+cu121` pin**（AGENTS 约束），PKG-1 只在 `setup.py` 侧剥离。
- **不把 slow 测试塞进 PR CI**（#7 的"可以"按非阻塞慢测执行）。
- `keepGUIAlive` 死代码、bundle 分块优化：留待后续，不进本计划。

---

## 4. 待验证（写代码时先确认，禁止臆测）

1. pydantic v2 lax 模式对 `clicks` 小数坐标是截断还是 422（决定 D-2 是否还需要 handler 侧兜底）。
2. cv2 PNG `IMWRITE_PNG_COMPRESSION=6` 对 12MP RGBA mask 的编码耗时（D-5；PIL 数据已有，cv2 需实测）。
3. `SD-17` 删除后 `test_brushnet` 其余断言是否依赖模糊 mask（删 `:96` 后跑一遍）。

## 5. 验收与顺序

1. Phase 1 → `bash scripts/check.sh`（pytest fast + eslint + build + dist 拷贝）全绿；
2. 用**空闲随机端口**起独立实例（cv2 模型，绝不碰生产环境），跑 D-4/D-5 的并发与耗时基准，对照本文件内的实测基线（12MP PNG 编码 7247ms → 目标 ≈1.2s；`/samplers` 并发 max 4696ms → 目标 <100ms）；
3. Phase 2 → 新增单测（D-1、D-9、D-15、D-17）全绿；
4. Phase 3 → 收尾，`git status` 仅包含计划内文件；
5. 每个 Phase 一次独立 commit，出问题可单点回滚。
