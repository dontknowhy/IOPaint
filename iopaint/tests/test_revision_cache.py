"""Revision cache：命中、回退、逐出、admission 与补丁位精确。"""
from typing import Any, Optional, Tuple

import numpy as np
import pytest

from iopaint.revision_cache import KEYFRAME_INTERVAL, Lease, RevisionCache, sha256_bytes


def _frame(value: int = 0, size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    return np.full((size[0], size[1], 3), value, dtype=np.uint8)


def _edit(frame: np.ndarray, box: Tuple[int, int, int, int], value: int) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = box
    out[y1:y2, x1:x2] = value
    return out


def test_hash_hit_and_revision_hit() -> None:
    cache = RevisionCache(max_bytes=10 * 1024 * 1024)
    frame = _frame(10)
    digest = sha256_bytes(b"round0")
    entry = cache.create(frame, None, "jpeg", {}, 2, b"round0", seed_hash=digest)
    assert entry is not None

    lease = cache.begin_by_hash(digest)
    assert lease is not None
    assert lease.image.shape == frame.shape
    rev = lease.commit(_edit(frame, (4, 4, 12, 12), 200))
    lease.release()
    assert rev is not None

    lease2 = cache.begin_by_revision(rev)
    assert lease2 is not None
    assert lease2.image[5, 5, 0] == 200
    lease2.release()
    assert cache.stats["hits_hash"] == 1
    assert cache.stats["hits_revision"] == 1


def test_old_revision_rewind_is_bit_exact_and_branches() -> None:
    cache = RevisionCache(max_bytes=10 * 1024 * 1024)
    base = _frame(0)
    entry = cache.create(base.copy(), None, "png", {}, None, None, seed_hash="h0")
    assert entry is not None
    lease: Optional[Lease] = cache.begin_entry(entry)
    assert lease is not None
    r1 = lease.commit(_edit(base, (0, 0, 8, 8), 50))
    lease.release()
    assert r1 is not None

    lease = cache.begin_by_revision(r1)
    assert lease is not None
    lease.release()

    # 在 r1 上继续两轮
    lease = cache.begin_by_revision(r1)
    assert lease is not None
    f2 = _edit(lease.image, (8, 0, 16, 8), 100)
    r2 = lease.commit(f2)
    lease.release()
    assert r2 is not None
    lease = cache.begin_by_revision(r2)
    assert lease is not None
    f3 = _edit(lease.image, (0, 8, 8, 16), 150)
    r3 = lease.commit(f3)
    lease.release()
    assert r3 is not None

    # 导出旧 revision 不改变 canonical
    snap = cache.snapshot(r1)
    assert snap is not None
    assert np.array_equal(snap[0], _edit(base, (0, 0, 8, 8), 50))
    snap3 = cache.snapshot(r3)
    assert snap3 is not None
    assert np.array_equal(snap3[0], f3)

    # 从 r1 分支继续编辑：r2/r3 失效，r1 状态作为新起点
    lease = cache.begin_by_revision(r1)
    assert lease is not None
    assert np.array_equal(lease.image, _edit(base, (0, 0, 8, 8), 50))
    r4 = lease.commit(_edit(lease.image, (16, 16, 24, 24), 255))
    lease.release()
    assert r4 is not None
    assert cache.begin_by_revision(r2) is None
    assert cache.begin_by_revision(r3) is None
    assert cache.begin_by_revision(r4) is not None


def test_keyframe_depth_is_bounded() -> None:
    cache = RevisionCache(max_bytes=10 * 1024 * 1024)
    base = _frame(0)
    entry = cache.create(base.copy(), None, "png", {}, None, None)
    assert entry is not None
    revisions: list[str] = []
    lease = cache.begin_entry(entry)
    for i in range(12):
        target = _edit(lease.image, (i, i, i + 2, i + 2), 200)
        rev = lease.commit(target)
        assert rev is not None
        revisions.append(rev)
    lease.release()
    assert len(entry.steps) <= KEYFRAME_INTERVAL
    # 最老的 revision 已被 keyframe 收缩，当前 revision 仍可命中
    assert cache.begin_by_revision(revisions[0]) is None
    assert cache.begin_by_revision(revisions[-1]) is not None


def test_empty_diff_does_not_add_generation() -> None:
    cache = RevisionCache(max_bytes=10 * 1024 * 1024)
    base = _frame(1)
    entry = cache.create(base.copy(), None, "png", {}, None, None)
    assert entry is not None
    lease = cache.begin_entry(entry)
    rev0 = lease.commit(base)
    lease.release()
    assert entry.generation == 0
    assert rev0 is not None


def test_shape_change_resets_to_new_keyframe() -> None:
    cache = RevisionCache(max_bytes=10 * 1024 * 1024)
    base = _frame(0, (16, 16))
    entry = cache.create(base.copy(), None, "png", {}, None, None)
    assert entry is not None
    lease = cache.begin_entry(entry)
    bigger = _frame(9, (24, 24))
    rev = lease.commit(bigger)
    lease.release()
    assert entry.generation == 1
    assert entry.steps == []
    snap = cache.snapshot(rev) if rev is not None else None
    assert snap is not None
    assert np.array_equal(snap[0], bigger)


def test_lru_eviction_and_miss_reason() -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    cache = RevisionCache(max_bytes=frame.nbytes + 64)
    first = cache.create(frame.copy(), None, "png", {}, None, None, seed_hash="a")
    assert first is not None
    lease = cache.begin_by_hash("a")
    assert lease is not None
    rev = lease.commit(frame.copy())
    lease.release()
    assert rev is not None
    second = cache.create(frame.copy(), None, "png", {}, None, None, seed_hash="b")
    assert second is not None
    # 第一条链（含 a/rev）被逐出
    assert cache.begin_by_hash("a") is None
    assert cache.begin_by_revision(rev) is None
    assert cache.stats["evictions"] >= 1


def test_admission_thresholds() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)  # 192B
    history_cache = RevisionCache(max_bytes=frame.nbytes * 4)
    assert (
        history_cache.create(frame.copy(), None, "png", {}, None, None) is not None
    )
    assert history_cache.stats["admitted_history"] == 1

    single_cache = RevisionCache(max_bytes=int(frame.nbytes * 1.5))
    entry = single_cache.create(frame.copy(), None, "png", {}, None, None)
    assert entry is not None and entry.single_entry
    assert single_cache.stats["admitted_single_entry"] == 1

    disabled_cache = RevisionCache(max_bytes=frame.nbytes // 2)
    assert disabled_cache.create(frame.copy(), None, "png", {}, None, None) is None
    assert disabled_cache.stats["rejected_no_cache"] == 1


def test_single_entry_keeps_latest_only() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    cache = RevisionCache(max_bytes=int(frame.nbytes * 1.5))
    entry = cache.create(frame.copy(), None, "png", {}, None, None)
    assert entry is not None
    lease = cache.begin_entry(entry)
    r1 = lease.commit(_edit(frame, (0, 0, 4, 4), 100))
    r2 = lease.commit(_edit(lease.image, (4, 4, 8, 8), 200))
    lease.release()
    assert r1 is not None and r2 is not None
    assert r1 != r2
    assert cache.begin_by_revision(r1) is None
    assert cache.begin_by_revision(r2) is not None


def test_disabled_cache() -> None:
    cache = RevisionCache(max_bytes=0)
    assert cache.create(_frame(), None, "png", {}, None, None) is None
    assert cache.stats["rejected_no_cache"] == 1


def test_delta_compress_roundtrip_and_memory_accounting() -> None:
    pytest.importorskip("zstandard")
    base = np.zeros((96, 96, 3), dtype=np.uint8)
    base[:, :48] = 40
    raw = RevisionCache(max_bytes=10 * 1024 * 1024)
    comp = RevisionCache(max_bytes=10 * 1024 * 1024, delta_compress=True)

    snapshots = []
    entries = []
    for cache in (raw, comp):
        entry = cache.create(base.copy(), None, "png", {}, None, None)
        assert entry is not None
        entries.append(entry)
        lease: Optional[Lease] = cache.begin_entry(entry)
        assert lease is not None
        r1 = lease.commit(_edit(base, (8, 8, 72, 72), 200))
        lease.release()
        assert r1 is not None
        lease = cache.begin_by_revision(r1)
        assert lease is not None
        r2 = lease.commit(_edit(lease.image, (16, 16, 48, 48), 90))
        lease.release()
        assert r2 is not None
        snap = cache.snapshot(r1)
        assert snap is not None
        snapshots.append((snap[0].copy(), entry.nbytes))
        assert cache.snapshot(r2) is not None

    assert np.array_equal(snapshots[0][0], snapshots[1][0])
    # 补丁是高度可压缩的平坦色块，压缩后历史占用应明显更小
    assert snapshots[1][1] < snapshots[0][1]
    comp_entry = entries[1]
    assert comp_entry.steps
    assert all(step.compressed is not None for step in comp_entry.steps)


def test_alias_keeps_generation_of_response_bytes() -> None:
    cache = RevisionCache(max_bytes=10 * 1024 * 1024)
    base = _frame(0)
    entry = cache.create(base.copy(), None, "png", {}, None, None)
    assert entry is not None
    lease: Optional[Lease] = cache.begin_entry(entry)
    assert lease is not None
    r1 = lease.commit(_edit(base, (0, 0, 4, 4), 10))
    lease.release()
    assert r1 is not None
    cache.alias("resp1", r1)
    lease = cache.begin_by_revision(r1)
    assert lease is not None
    r2 = lease.commit(_edit(lease.image, (4, 4, 8, 8), 20))
    lease.release()
    assert r2 is not None

    # 旧响应字节的别名仍指向它产生时的 generation
    lease = cache.begin_by_hash("resp1")
    assert lease is not None
    assert np.array_equal(lease.image, _edit(base, (0, 0, 4, 4), 10))
    assert lease.current_revision == r1
    lease.release()
    # 从旧 hash 进入 = 分支到那一轮的 generation，更新的 revision 失效
    assert cache.begin_by_revision(r2) is None


def test_shape_change_commit_stores_rebuilt_alpha_and_accounts_bytes() -> None:
    """画布变化时 alpha 只能由调用方重建后传入，不能再被 resize 或丢弃。"""
    cache = RevisionCache(max_bytes=10 * 1024 * 1024)
    base = _frame(0, (8, 8))
    alpha = np.tile(np.arange(8, dtype=np.uint8), (8, 1))
    entry = cache.create(base.copy(), alpha, "png", {}, None, None)
    assert entry is not None
    lease = cache.begin_entry(entry)
    assert lease is not None
    new_alpha = np.full((8, 12), 255, dtype=np.uint8)
    rev = lease.commit(_frame(9, (8, 12)), alpha=new_alpha)
    lease.release()
    assert rev is not None
    snap = cache.snapshot(rev)
    assert snap is not None
    assert snap[1] is not None
    assert np.array_equal(snap[1], new_alpha)
    # alpha 的字节数也计入预算：_bytes 必须与条目实际占用一致
    assert cache.total_bytes == entry.nbytes


def test_rewind_refunds_popped_step_bytes() -> None:
    """回退弹掉的撤销补丁必须从 _bytes 里扣掉，否则反复分支会虚高预算。"""
    frame = _frame(0, (32, 32))
    cache = RevisionCache(max_bytes=7000)
    entry = cache.create(frame.copy(), None, "png", {}, None, None)
    assert entry is not None
    rev0 = next(rev for rev, gen in entry.revisions.items() if gen == 0)
    assert cache.total_bytes == entry.nbytes
    for _ in range(5):
        lease = cache.begin_entry(entry)
        assert lease is not None
        r1 = lease.commit(_edit(lease.image, (0, 0, 8, 8), 50))
        lease.release()
        assert r1 is not None
        lease = cache.begin_by_revision(r1)
        assert lease is not None
        r2 = lease.commit(_edit(lease.image, (8, 8, 16, 16), 100))
        lease.release()
        assert r2 is not None
        assert cache.total_bytes == sum(e.nbytes for e in cache._entries.values())
        lease = cache.begin_by_revision(rev0)
        assert lease is not None
        lease.release()
        assert entry.steps == []
        assert cache.total_bytes == entry.nbytes


def test_begin_entry_on_evicted_entry_returns_none() -> None:
    """create() 与 begin_entry() 之间被逐出的条目不能再碰 LRU（KeyError → 500）。"""
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    cache = RevisionCache(max_bytes=800)
    first = cache.create(frame.copy(), None, "png", {}, None, None)
    assert first is not None
    # 小预算：第二次 create 会在 first.pins == 0 时把它逐出
    second = cache.create(frame.copy(), None, "png", {}, None, None, seed_hash="b")
    assert second is not None
    assert cache.begin_entry(first) is None
    lease = cache.begin_entry(second)
    assert lease is not None
    lease.release()


def test_commit_holds_entry_lock_while_trimming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_trim_locked 会 pop exclude.steps，_commit 必须自己持有 entry.lock。

    不能依赖调用方（Lease.commit）恰好持锁——直接调 _commit 的路径同样要安全。
    """
    cache = RevisionCache(max_bytes=10 * 1024 * 1024)
    base = _frame(0)
    entry = cache.create(base.copy(), None, "png", {}, None, None)
    assert entry is not None
    real_trim = RevisionCache._trim_locked
    owned: list[bool] = []

    def probe(self: RevisionCache, exclude: Any = None) -> None:
        owned.append(bool(getattr(entry.lock, "_is_owned")()))
        real_trim(self, exclude=exclude)

    monkeypatch.setattr(RevisionCache, "_trim_locked", probe)
    assert cache._commit(entry, _edit(base, (0, 0, 8, 8), 100)) is not None
    assert owned == [True]


def test_rewind_failure_releases_lock_and_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """补丁材质化抛异常时必须释放 entry.lock 与 pin，否则该 revision 永久阻塞。"""
    cache = RevisionCache(max_bytes=10 * 1024 * 1024)
    base = _frame(0)
    entry = cache.create(base.copy(), None, "png", {}, None, None)
    assert entry is not None
    rev0 = next(rev for rev, gen in entry.revisions.items() if gen == 0)
    lease = cache.begin_entry(entry)
    assert lease is not None
    r1 = lease.commit(_edit(base, (0, 0, 8, 8), 50))
    lease.release()
    assert r1 is not None
    lease = cache.begin_by_revision(r1)
    assert lease is not None
    r2 = lease.commit(_edit(lease.image, (8, 0, 16, 8), 100))
    lease.release()
    assert r2 is not None

    real_rewind = RevisionCache._rewind_locked

    def boom(self: RevisionCache, entry: Any, generation: int) -> bool:
        raise RuntimeError("corrupt undo patch")

    monkeypatch.setattr(RevisionCache, "_rewind_locked", boom)
    with pytest.raises(RuntimeError, match="corrupt"):
        cache.begin_by_revision(rev0)
    assert entry.pins == 0
    assert not getattr(entry.lock, "_is_owned")()
    # 异常不应影响更早 revision 的正常回退
    monkeypatch.setattr(RevisionCache, "_rewind_locked", real_rewind)
    lease = cache.begin_by_revision(r1)
    assert lease is not None
    lease.release()
