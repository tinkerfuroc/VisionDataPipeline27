from vsp.labeling.ranges import add_range, subtract_range


def test_add_to_empty():
    assert add_range([], 3, 7) == [(3, 7)]


def test_add_merges_overlapping_and_adjacent():
    assert add_range([(1, 4)], 5, 9) == [(1, 9)]  # adjacent
    assert add_range([(1, 4), (10, 12)], 3, 10) == [(1, 12)]  # bridges both
    assert add_range([(1, 4)], 6, 9) == [(1, 4), (6, 9)]  # gap of one frame stays separate


def test_add_accepts_reversed_bounds():
    assert add_range([], 9, 5) == [(5, 9)]


def test_add_keeps_result_sorted():
    assert add_range([(20, 25)], 1, 3) == [(1, 3), (20, 25)]


def test_subtract_middle_splits():
    assert subtract_range([(1, 10)], 4, 6) == [(1, 3), (7, 10)]


def test_subtract_edges_and_whole():
    assert subtract_range([(1, 10)], 1, 3) == [(4, 10)]
    assert subtract_range([(1, 10)], 8, 99) == [(1, 7)]
    assert subtract_range([(1, 10)], 0, 99) == []


def test_subtract_untouched_and_multiple():
    assert subtract_range([(1, 3), (8, 9)], 5, 6) == [(1, 3), (8, 9)]
    assert subtract_range([(1, 5), (7, 12)], 4, 8) == [(1, 3), (9, 12)]
