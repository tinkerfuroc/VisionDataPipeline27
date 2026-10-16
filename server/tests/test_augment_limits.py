from vdp.augment.limits import plan_generation


def test_no_limit_or_room_enough_keeps_everything():
    p = plan_generation(10, 3, 2, None)
    assert p.composites == list(range(30)) and len(p.lighting) == 20 and not p.capped
    assert plan_generation(10, 3, 2, 60).capped is False  # 10 + 30 + 20 == 60


def test_cap_splits_in_proportion_and_fills_the_room():
    p = plan_generation(798, 3, 2, 2000)  # room 1202, C 2394, L 1596
    assert len(p.composites) == round(1202 * 2394 / 3990) == 721
    assert len(p.lighting) == 1202 - 721 == 481 and p.capped
    assert p.composites == list(range(721))


def test_lighting_takes_whole_rounds_then_evenly_spaced_frames():
    p = plan_generation(10, 0, 3, 10 + 14)  # room 14: one full round (10) + 4 of the next
    assert p.lighting[:10] == [(i, 0) for i in range(10)]
    assert p.lighting[10:] == [(0, 1), (2, 1), (5, 1), (7, 1)]  # floor(j*10/4)


def test_one_kind_only_and_tiny_rooms():
    assert plan_generation(5, 2, 0, 8).composites == [0, 1, 2]
    assert plan_generation(5, 0, 1, 6).lighting == [(0, 0)]
    p = plan_generation(5, 3, 1, 6)  # room 1: rounds to all-composite
    assert len(p.composites) + len(p.lighting) == 1
