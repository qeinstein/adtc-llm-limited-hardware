"""Frozen-slice helper for staged MCQ eval (screen vs confirmation sets)."""

from scripts.mcq_eval import _take


def test_take_head_slice():
    assert _take([1, 2, 3, 4, 5], 3) == [1, 2, 3]


def test_take_disjoint_offsets_cover_without_overlap():
    items = list(range(10))
    screen = _take(items, 4, offset=0)
    confirm = _take(items, 4, offset=4)
    assert screen == [0, 1, 2, 3]
    assert confirm == [4, 5, 6, 7]
    assert not set(screen) & set(confirm)


def test_take_clamps_at_end_and_negative_offset():
    assert _take([1, 2, 3], 5, offset=1) == [2, 3]
    assert _take([1, 2, 3], 2, offset=-4) == [1, 2]
