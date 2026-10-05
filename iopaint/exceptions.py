class ModelLoadError(RuntimeError):
    """模型下载/加载失败。

    原先这里 `raise SystemExit(1)` / `exit(-1)`：如果发生在 API 请求路径上
    （例如运行期 `switch_model` 触发下载失败），会直接杀掉整个服务进程，
    且 SystemExit 是 BaseException，绕过 FastAPI 的异常处理。

    改成 RuntimeError 子类后：请求路径 → 由 `api.py` 的异常中间件转成 500；
    启动路径（`cli.py start`）→ 捕获后 `SystemExit(1)`，启动期语义不变。
    """
