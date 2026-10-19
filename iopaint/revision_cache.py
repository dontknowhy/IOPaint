"""进程内无损编辑状态（revision cache）。

盒子契约：服务端内存里的解码后像素是 canonical state，
每轮 inpaint 直接在它上面继续编辑，不再把上一轮的**有损响应字节**解码回来。
本模块只负责保存这份 canonical 与它的撤销补丁，不做任何编解码。

内存布局：
- 每条链只有**一份**整帧 ``image``（当前 canonical）；
- 历史按“绝对补丁”保存：每一步记录被改写区域**改写前**的像素，
  撤销时按相反顺序贴回即可，不需要第二份整帧，也没有全帧拷贝；
- 每 8 步丢弃最老的补丁（keyframe 收缩），因此最老可达 revision 会失效，
  但当前 canonical 始终完整。

线程模型：全局锁只保护 LRU / revision 表；每张图另有自己的锁，推理期间
只 pin 住条目（避免被逐出），不会把编码/IO 排进 ``ModelManager.lock``。
"""
from __future__ import annotations

import hashlib
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

try:  # 可选依赖：只有 --delta-compress 开启时才需要
    import zstandard as _zstd
except ImportError:  # pragma: no cover - 环境差异
    _zstd = None  # type: ignore[assignment]

# keyframe 周期。保留最近 K 个撤销补丁，更老的 revision 失效。
KEYFRAME_INTERVAL = 8

Box = Tuple[int, int, int, int]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class _Step:
    """一次编辑的撤销补丁（绝对内容，不是差分）。"""

    generation: int
    box: Box
    before: Optional[np.ndarray] = None
    compressed: Optional[bytes] = None
    shape: Tuple[int, ...] = ()
    dtype: str = "uint8"

    @property
    def nbytes(self) -> int:
        if self.compressed is not None:
            return len(self.compressed)
        return 0 if self.before is None else int(self.before.nbytes)

    def materialize(self) -> np.ndarray:
        if self.before is not None:
            return self.before
        if self.compressed is None or _zstd is None:
            raise RuntimeError("compressed undo patch without zstandard")
        raw = _zstd.decompress(self.compressed)
        return np.frombuffer(raw, dtype=np.dtype(self.dtype)).reshape(self.shape)


@dataclass
class _Entry:
    image: np.ndarray
    alpha: Optional[np.ndarray]
    ext: str
    infos: Dict
    source_sampling: Optional[int]
    source_bytes: Optional[bytes]
    # 掩膜外 DCT 导出用：以 8x8 luma 块为粒度的“曾经改过”位图
    changed_blocks: np.ndarray
    generation: int = 0
    floor: int = 0
    steps: list[_Step] = field(default_factory=list)
    revisions: Dict[str, int] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)
    pins: int = 0
    last_used: float = field(default_factory=time.monotonic)
    single_entry: bool = False

    @property
    def nbytes(self) -> int:
        total = self.image.nbytes
        if self.alpha is not None:
            total += self.alpha.nbytes
        total += self.changed_blocks.nbytes
        if self.source_bytes is not None:
            total += len(self.source_bytes)
        for step in self.steps:
            total += step.nbytes
        return total


def _changed_block_mask(diff: np.ndarray) -> np.ndarray:
    """把逐像素变化图缩成 8x8 块粒度（不完整边缘块按“变化”处理）。"""
    h, w = diff.shape[:2]
    bh = (h + 7) // 8
    bw = (w + 7) // 8
    out = np.zeros((bh, bw), dtype=bool)
    hh, ww = bh * 8, bw * 8
    if hh != h or ww != w:
        padded = np.zeros((hh, ww), dtype=bool)
        padded[:h, :w] = diff
        diff = padded
    out[:] = diff.reshape(bh, 8, bw, 8).any(axis=(1, 3))
    return out


def _bbox_of_diff(diff: np.ndarray) -> Optional[Box]:
    rows = np.flatnonzero(diff.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(diff.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


class Lease:
    """一次 inpaint 对某条链的独占租约。

    ``image`` 是 canonical 的可写视图；命中旧 revision 时已经在锁内回退到
    该 revision（分支语义）。``commit`` 把推理结果写回并生成新 revision。
    """

    def __init__(self, cache: "RevisionCache", entry: _Entry):
        self.cache = cache
        self.entry = entry
        self._committed = False

    @property
    def image(self) -> np.ndarray:
        return self.entry.image

    @property
    def alpha(self) -> Optional[np.ndarray]:
        return self.entry.alpha

    @property
    def ext(self) -> str:
        return self.entry.ext

    @property
    def infos(self) -> Dict:
        return self.entry.infos

    @property
    def source_sampling(self) -> Optional[int]:
        return self.entry.source_sampling

    @property
    def source_bytes(self) -> Optional[bytes]:
        return self.entry.source_bytes

    @property
    def changed_blocks(self) -> np.ndarray:
        return self.entry.changed_blocks

    @property
    def current_revision(self) -> Optional[str]:
        for rev, gen in self.entry.revisions.items():
            if gen == self.entry.generation:
                return rev
        return None

    def commit(
        self,
        result_rgb: np.ndarray,
        *,
        keep_unmasked_area: bool = True,
    ) -> Optional[str]:
        """提交一轮结果，返回新 revision。未采纳缓存时返回 None。"""
        rev = self.cache._commit(self.entry, result_rgb)
        self._committed = True
        return rev

    def branch_generation(self) -> int:
        return self.entry.generation

    def release(self) -> None:
        if not self._committed:
            # 推理失败：保持当前（可能已回退到的）状态，只释放 pin。
            pass
        self.cache._release(self.entry)


class RevisionCache:
    def __init__(
        self, max_bytes: int = 512 * 1024 * 1024, delta_compress: bool = False
    ):
        self.max_bytes = max(0, int(max_bytes))
        self.delta_compress = bool(delta_compress and _zstd is not None)
        if delta_compress and _zstd is None:
            logger.warning(
                "--delta-compress requested but zstandard is not installed; "
                "undo patches stay raw"
            )
        self._lock = threading.RLock()
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        self._revisions: Dict[str, _Entry] = {}
        self._aliases: "OrderedDict[str, Tuple[_Entry, int]]" = OrderedDict()
        self._evicted_hashes: "OrderedDict[str, None]" = OrderedDict()
        self._ever_created = 0
        self._bytes = 0
        # 可观测指标（测试与 /server-config 之外的排障用）
        self.stats = {
            "hits_revision": 0,
            "hits_hash": 0,
            "miss_first_import": 0,
            "miss_evicted": 0,
            "miss_restart": 0,
            "admitted_history": 0,
            "admitted_single_entry": 0,
            "rejected_no_cache": 0,
            "evictions": 0,
        }

    # ------------------------------------------------------------------
    # 查询 / admission
    # ------------------------------------------------------------------
    def _remember_evicted(self, token: str) -> None:
        if not token:
            return
        self._evicted_hashes[token] = None
        self._evicted_hashes.move_to_end(token)
        while len(self._evicted_hashes) > 1024:
            self._evicted_hashes.popitem(last=False)

    def _miss_reason(self, token: Optional[str]) -> str:
        if token and token in self._evicted_hashes:
            return "evicted"
        if self._ever_created == 0:
            return "restart"
        return "restart"

    def _lookup_locked(
        self, token: Optional[str], by_hash: bool = False
    ) -> Optional[Tuple[_Entry, int]]:
        if not token:
            return None
        if by_hash:
            hit = self._aliases.get(token)
            if hit is None:
                return None
            entry, gen = hit
            if gen < entry.floor or gen > entry.generation:
                return None
        else:
            found = self._revisions.get(token)
            if found is None:
                return None
            gen_opt = found.revisions.get(token)
            if gen_opt is None or gen_opt < found.floor:
                return None
            entry, gen = found, gen_opt
        entry.last_used = time.monotonic()
        self._entries.move_to_end(str(id(entry)))
        return entry, gen

    def _begin(self, token: Optional[str], by_hash: bool) -> Optional[Lease]:
        with self._lock:
            hit = self._lookup_locked(token, by_hash=by_hash)
            if hit is None:
                reason = self._miss_reason(token)
                if token:
                    key = "miss_evicted" if reason == "evicted" else "miss_restart"
                    self.stats[key] += 1
                    logger.info(
                        f"revision cache miss ({reason}): "
                        f"{'hash' if by_hash else 'revision'}={token[:16]}…"
                    )
                return None
            entry, gen = hit
            entry.pins += 1
        entry.lock.acquire()
        # 回退到请求的 generation（分支语义）。单条目模式只能是最新。
        if gen != entry.generation:
            if not self._rewind_locked(entry, gen):
                entry.lock.release()
                with self._lock:
                    entry.pins -= 1
                logger.info(
                    f"revision cache miss (evicted): revision={str(token)[:16]}…"
                )
                self.stats["miss_evicted"] += 1
                return None
        if by_hash:
            self.stats["hits_hash"] += 1
        else:
            self.stats["hits_revision"] += 1
        return Lease(self, entry)

    def begin_by_revision(self, revision: str) -> Optional[Lease]:
        return self._begin(revision, by_hash=False)

    def begin_by_hash(self, digest: str) -> Optional[Lease]:
        return self._begin(digest, by_hash=True)

    def begin_entry(self, entry: _Entry) -> Lease:
        """为刚 create() 的条目取租约（首轮导入路径）。"""
        with self._lock:
            entry.pins += 1
            entry.last_used = time.monotonic()
            self._entries.move_to_end(str(id(entry)))
        entry.lock.acquire()
        return Lease(self, entry)

    def _release(self, entry: _Entry) -> None:
        try:
            entry.lock.release()
        finally:
            with self._lock:
                entry.pins = max(0, entry.pins - 1)

    def _rewind_locked(self, entry: _Entry, generation: int) -> bool:
        if generation < entry.floor or generation > entry.generation:
            return False
        # steps 覆盖 (floor, generation]；按 generation 倒序贴回 before
        while entry.steps and entry.steps[-1].generation > generation:
            step = entry.steps.pop()
            x1, y1, x2, y2 = step.box
            entry.image[y1:y2, x1:x2] = step.materialize()
        entry.generation = generation
        # 失效的 revision 清掉
        for rev, gen in list(entry.revisions.items()):
            if gen > generation:
                entry.revisions.pop(rev, None)
        return True

    # ------------------------------------------------------------------
    # 创建 / 提交
    # ------------------------------------------------------------------
    def create(
        self,
        image: np.ndarray,
        alpha: Optional[np.ndarray],
        ext: str,
        infos: Dict,
        source_sampling: Optional[int],
        source_bytes: Optional[bytes],
        seed_hash: Optional[str] = None,
    ) -> Optional[_Entry]:
        if self.max_bytes <= 0:
            self.stats["rejected_no_cache"] += 1
            logger.warning(
                "revision cache disabled (--revision-cache-mb 0): "
                "each round falls back to the legacy lossy path"
            )
            return None

        base_bytes = int(image.nbytes)
        single_entry = False
        if base_bytes > self.max_bytes:
            self.stats["rejected_no_cache"] += 1
            logger.warning(
                f"revision cache disabled: canonical frame {base_bytes / 2**20:.0f}MB "
                f"> budget {self.max_bytes / 2**20:.0f}MB, cache disabled for "
                f"{image.shape[1]}x{image.shape[0]} (use PNG source or larger "
                f"--revision-cache-mb)"
            )
            return None
        if base_bytes > self.max_bytes // 2:
            single_entry = True
            self.stats["admitted_single_entry"] += 1
            logger.warning(
                f"revision cache single-entry: canonical frame {base_bytes / 2**20:.0f}MB "
                f"> budget/2, keep latest canonical only (no undo history)"
            )
        else:
            self.stats["admitted_history"] += 1

        changed = np.zeros(
            ((int(image.shape[0]) + 7) // 8, (int(image.shape[1]) + 7) // 8),
            dtype=bool,
        )
        entry = _Entry(
            image=np.ascontiguousarray(image),
            alpha=alpha,
            ext=ext,
            infos=dict(infos or {}),
            source_sampling=source_sampling,
            source_bytes=source_bytes,
            changed_blocks=changed,
            single_entry=single_entry,
        )
        rev = secrets.token_hex(8)
        entry.revisions[rev] = 0
        key = str(id(entry))
        with self._lock:
            if not self._make_room_locked(entry.nbytes):
                self.stats["rejected_no_cache"] += 1
                logger.warning(
                    "revision cache disabled: no room for canonical frame "
                    f"({entry.nbytes / 2**20:.0f}MB needed)"
                )
                return None
            self._entries[key] = entry
            self._revisions[rev] = entry
            if seed_hash:
                self._aliases[seed_hash] = (entry, 0)
            self._bytes += entry.nbytes
            self._ever_created += 1
        return entry

    def alias(self, digest: str, revision: str) -> None:
        """把“响应字节的 sha256”映射到 revision（无前端改动也能命中）。"""
        with self._lock:
            entry = self._revisions.get(revision)
            if entry is None or not digest:
                return
            gen = entry.revisions.get(revision)
            if gen is None:
                return
            self._aliases[digest] = (entry, gen)
            self._aliases.move_to_end(digest)
            # 旧别名大量堆积时清掉最老的，保持 LRU 结构有界
            while len(self._aliases) > 4096:
                old, _ = self._aliases.popitem(last=False)
                if old in self._evicted_hashes:
                    self._evicted_hashes.pop(old, None)

    def _new_step(self, generation: int, box: Box, patch: np.ndarray) -> _Step:
        if self.delta_compress:
            assert _zstd is not None
            return _Step(
                generation=generation,
                box=box,
                compressed=_zstd.compress(patch.tobytes(), 3),
                shape=patch.shape,
                dtype=patch.dtype.str,
            )
        return _Step(generation=generation, box=box, before=patch)

    def _commit(self, entry: _Entry, result_rgb: np.ndarray) -> Optional[str]:
        with entry.lock:
            old_cost = entry.nbytes
            if result_rgb.shape[:2] != entry.image.shape[:2]:
                # 裁剪/扩展（outpainting）改变了画布：整帧成为新 keyframe。
                entry.image = np.ascontiguousarray(result_rgb)
                entry.alpha = None
                entry.steps.clear()
                entry.generation += 1
                entry.floor = entry.generation
                entry.revisions.clear()
                entry.changed_blocks = np.ones(
                    (
                        (result_rgb.shape[0] + 7) // 8,
                        (result_rgb.shape[1] + 7) // 8,
                    ),
                    dtype=bool,
                )
            else:
                diff = np.any(result_rgb != entry.image, axis=2)
                box = _bbox_of_diff(diff) if diff.any() else None
                if box is None:
                    # 空掩膜 / 无变化：不新增 generation，直接复用当前 revision
                    for rev, gen in entry.revisions.items():
                        if gen == entry.generation:
                            return rev
                    return None
                x1, y1, x2, y2 = box
                entry.steps.append(
                    self._new_step(
                        generation=entry.generation + 1,
                        box=box,
                        patch=entry.image[y1:y2, x1:x2].copy(),
                    )
                )
                entry.image[y1:y2, x1:x2] = result_rgb[y1:y2, x1:x2]
                entry.generation += 1
                entry.changed_blocks |= _changed_block_mask(diff)

            if entry.single_entry:
                # 单条目模式：只保留最新 canonical，撤销历史不保留。
                entry.steps.clear()
                entry.floor = entry.generation
                entry.revisions.clear()
            elif len(entry.steps) > KEYFRAME_INTERVAL:
                drop = len(entry.steps) - KEYFRAME_INTERVAL
                for _ in range(drop):
                    entry.steps.pop(0)
                entry.floor = entry.generation - len(entry.steps)
                for rev, gen in list(entry.revisions.items()):
                    if gen < entry.floor:
                        entry.revisions.pop(rev, None)

            rev = secrets.token_hex(8)
            entry.revisions[rev] = entry.generation
            entry.last_used = time.monotonic()

        # LRU 记账（在 entry.lock 之外拿全局锁，避免与逐出死锁）
        with self._lock:
            self._bytes += entry.nbytes - old_cost
            self._revisions[rev] = entry
            self._entries.move_to_end(str(id(entry)))
            self._trim_locked(exclude=entry)
        return rev

    # ------------------------------------------------------------------
    # 导出视图
    # ------------------------------------------------------------------
    def snapshot(
        self, revision: str
    ) -> Optional[
        Tuple[
            np.ndarray,
            Optional[np.ndarray],
            str,
            Dict,
            Optional[int],
            Optional[bytes],
            Optional[np.ndarray],
        ]
    ]:
        """返回 revision 对应的像素副本（导出不回流，不会影响 canonical）。"""
        with self._lock:
            hit = self._lookup_locked(revision, by_hash=False)
            if hit is None:
                return None
            entry, gen = hit
            entry.pins += 1
        entry.lock.acquire()
        try:
            # 先复制当前 canonical，再在副本上按相反顺序贴回 before。
            # 这样旧 revision 的导出不会把缓存原地改回过去。
            image = entry.image.copy()
            for step in reversed(entry.steps):
                if step.generation <= gen:
                    break
                x1, y1, x2, y2 = step.box
                image[y1:y2, x1:x2] = step.materialize()
            return (
                image,
                None if entry.alpha is None else entry.alpha.copy(),
                entry.ext,
                dict(entry.infos),
                entry.source_sampling,
                entry.source_bytes,
                entry.changed_blocks.copy(),
            )
        finally:
            entry.lock.release()
            with self._lock:
                entry.pins = max(0, entry.pins - 1)

    # ------------------------------------------------------------------
    # LRU
    # ------------------------------------------------------------------
    def _make_room_locked(self, needed: int) -> bool:
        if needed > self.max_bytes:
            return False
        while self._bytes + needed > self.max_bytes and self._entries:
            evicted = False
            for key, entry in list(self._entries.items()):
                if entry.pins:
                    continue
                self._entries.pop(key, None)
                self._bytes -= entry.nbytes
                for rev in list(entry.revisions):
                    self._revisions.pop(rev, None)
                    self._remember_evicted(rev)
                for digest, (owner, _gen) in list(self._aliases.items()):
                    if owner is entry:
                        self._aliases.pop(digest, None)
                        self._remember_evicted(digest)
                self.stats["evictions"] += 1
                logger.info(
                    f"revision cache evicted entry "
                    f"({entry.nbytes / 2**20:.1f}MB, {entry.image.shape[1]}x"
                    f"{entry.image.shape[0]})"
                )
                evicted = True
                break
            if not evicted:
                return False
        return True

    def _trim_locked(self, exclude: Optional[_Entry] = None) -> None:
        while self._bytes > self.max_bytes and self._entries:
            evicted = False
            for key, entry in list(self._entries.items()):
                if entry is exclude or entry.pins:
                    continue
                self._entries.pop(key, None)
                self._bytes -= entry.nbytes
                for rev in list(entry.revisions):
                    self._revisions.pop(rev, None)
                    self._remember_evicted(rev)
                for digest, (owner, _gen) in list(self._aliases.items()):
                    if owner is entry:
                        self._aliases.pop(digest, None)
                        self._remember_evicted(digest)
                self.stats["evictions"] += 1
                logger.info(
                    f"revision cache evicted entry "
                    f"({entry.nbytes / 2**20:.1f}MB, {entry.image.shape[1]}x"
                    f"{entry.image.shape[0]})"
                )
                evicted = True
                break
            if not evicted:
                # 唯一剩余条目就是当前条目：收缩历史以腾空间
                if exclude is not None and exclude.steps:
                    step = exclude.steps.pop(0)
                    self._bytes -= step.nbytes
                    exclude.floor += 1
                    for rev, gen in list(exclude.revisions.items()):
                        if gen < exclude.floor:
                            exclude.revisions.pop(rev, None)
                    continue
                break

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._bytes
