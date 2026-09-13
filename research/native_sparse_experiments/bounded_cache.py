"""Small explicit byte-bounded cache primitive for selected expert records.

This module deliberately has no mmap dependency.  It models the transitional
runtime contract: a router supplies ``(layer, expert)`` keys, a cache miss
reads one packed record into a fixed aligned slot, and deterministic LRU
eviction keeps resident storage below a byte-derived slot budget.

It is single-threaded by design for batch-1 decode.  I/O scheduling and the
expert kernel are separate layers so they can be benchmarked independently.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterator


BundleKey = tuple[int, int]
Reader = Callable[[BundleKey], bytes]


@dataclass(frozen=True)
class CacheEntry:
    """Metadata for one resident fixed-size cache slot."""

    key: BundleKey
    slot: int
    payload_bytes: int


class BoundedExpertCache:
    """A deterministic LRU cache with fixed-size, byte-accounted slots.

    ``capacity_bytes`` is converted to a whole number of slots.  The unused
    remainder is never silently spent, and ``resident_bytes`` reports the
    actual fixed-slot reservation rather than the payload's logical size.
    ``slot_bytes`` should be the maximum packed record size rounded to the
    executor's alignment requirement.
    """

    def __init__(self, capacity_bytes: int, slot_bytes: int, *, alignment: int = 4096) -> None:
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")
        if slot_bytes <= 0:
            raise ValueError("slot_bytes must be positive")
        if alignment <= 0 or slot_bytes % alignment:
            raise ValueError("slot_bytes must be a positive multiple of alignment")
        self.capacity_bytes = capacity_bytes
        self.slot_bytes = slot_bytes
        self.slot_count = capacity_bytes // slot_bytes
        self.resident_bytes = self.slot_count * slot_bytes
        self._entries: OrderedDict[BundleKey, CacheEntry] = OrderedDict()
        self._payloads: dict[int, bytes] = {}
        self._free_slots = list(range(self.slot_count))
        self.requests = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.read_bytes = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: BundleKey) -> bool:
        return key in self._entries

    def keys(self) -> Iterator[BundleKey]:
        """Yield keys from oldest to newest, useful for deterministic tests."""

        return iter(self._entries)

    def _touch(self, key: BundleKey) -> bytes:
        entry = self._entries[key]
        self._entries.move_to_end(key)
        return self._payloads[entry.slot]

    def get(self, key: BundleKey) -> bytes | None:
        """Return a resident payload and account for a cache request."""

        self.requests += 1
        if key not in self._entries:
            self.misses += 1
            return None
        self.hits += 1
        return self._touch(key)

    def _evict_oldest(self) -> None:
        key, entry = self._entries.popitem(last=False)
        self._payloads.pop(entry.slot)
        self._free_slots.append(entry.slot)
        self.evictions += 1

    def _insert(self, key: BundleKey, payload: bytes) -> bytes:
        if len(payload) > self.slot_bytes:
            raise ValueError(
                f"payload for {key} is {len(payload)} bytes, exceeds slot_bytes={self.slot_bytes}"
            )
        if self.slot_count == 0:
            return payload
        while not self._free_slots:
            self._evict_oldest()
        slot = self._free_slots.pop(0)
        self._entries[key] = CacheEntry(key=key, slot=slot, payload_bytes=len(payload))
        self._payloads[slot] = bytes(payload)
        return self._payloads[slot]

    def load(self, key: BundleKey, reader: Reader) -> bytes:
        """Get ``key`` or read it once into a bounded slot on a miss."""

        cached = self.get(key)
        if cached is not None:
            return cached
        payload = bytes(reader(key))
        self.read_bytes += len(payload)
        return self._insert(key, payload)

    def clear(self) -> None:
        """Release every slot while retaining cumulative counters."""

        self._entries.clear()
        self._payloads.clear()
        self._free_slots = list(range(self.slot_count))

    def snapshot(self) -> dict[str, int | float]:
        """Return serializable accounting suitable for experiment artifacts."""

        return {
            "capacity_bytes": self.capacity_bytes,
            "slot_bytes": self.slot_bytes,
            "slot_count": self.slot_count,
            "resident_bytes": self.resident_bytes,
            "resident_payload_bytes": sum(entry.payload_bytes for entry in self._entries.values()),
            "entries": len(self._entries),
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / self.requests if self.requests else 0.0,
            "evictions": self.evictions,
            "read_bytes": self.read_bytes,
        }
