from __future__ import annotations

import pytest

from research.native_sparse_experiments.bounded_cache import (
    BoundedExpertCache,
    ExpertRecord,
    PreadExpertStore,
)


def test_fixed_slot_budget_and_deterministic_lru() -> None:
    reads: list[tuple[int, int]] = []

    def reader(key: tuple[int, int]) -> bytes:
        reads.append(key)
        return bytes([key[1]]) * 16

    cache = BoundedExpertCache(32, 16, alignment=1)
    assert cache.load((0, 1), reader) == bytes([1]) * 16
    assert cache.load((0, 2), reader) == bytes([2]) * 16
    assert cache.load((0, 1), reader) == bytes([1]) * 16
    assert cache.load((0, 3), reader) == bytes([3]) * 16
    assert list(cache.keys()) == [(0, 1), (0, 3)]
    assert reads == [(0, 1), (0, 2), (0, 3)]
    assert cache.snapshot() == {
        "capacity_bytes": 32, "slot_bytes": 16, "slot_count": 2,
        "resident_bytes": 32, "resident_payload_bytes": 32, "entries": 2,
        "requests": 4, "hits": 1, "misses": 3, "hit_rate": 0.25,
        "evictions": 1, "read_bytes": 48,
    }


def test_budget_remainder_is_not_used_and_zero_capacity_is_explicit() -> None:
    cache = BoundedExpertCache(31, 16, alignment=1)
    assert cache.slot_count == 1
    assert cache.resident_bytes == 16
    assert cache.load((0, 0), lambda _: b"payload") == b"payload"

    empty = BoundedExpertCache(15, 16, alignment=1)
    assert empty.slot_count == 0
    assert empty.load((0, 0), lambda _: b"payload") == b"payload"
    assert empty.snapshot()["resident_bytes"] == 0


def test_oversized_record_is_rejected_before_residency_can_grow() -> None:
    cache = BoundedExpertCache(16, 16, alignment=1)
    with pytest.raises(ValueError, match="exceeds"):
        cache.load((2, 7), lambda _: b"x" * 17)
    assert len(cache) == 0
    assert cache.resident_bytes == 16


def test_explicit_pread_store_feeds_fixed_cache_slots(tmp_path) -> None:
    path = tmp_path / "experts.bin"
    path.write_bytes(b"header" + b"A" * 16 + b"B" * 16)
    records = {
        (3, 4): ExpertRecord(offset=6, size=16),
        (3, 5): ExpertRecord(offset=22, size=16),
    }
    with PreadExpertStore(path, records) as store:
        cache = BoundedExpertCache(16, 16, alignment=1)
        assert cache.load((3, 4), store.read) == b"A" * 16
        assert cache.load((3, 5), store.read) == b"B" * 16
        assert store._fd >= 0
        assert cache.snapshot()["read_bytes"] == 32
