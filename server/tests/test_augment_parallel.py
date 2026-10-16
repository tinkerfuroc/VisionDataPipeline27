import os

import pytest
from vdp.augment.parallel import default_workers, run_units
from vdp.export import ExportCancelled


def square(x):
    return x * x


def boom(x):
    if x == 3:
        raise ValueError("unit 3 failed")
    return x


def pid(_):
    return os.getpid()


def test_results_in_unit_order_inline_and_pooled():
    for workers in (1, 3):
        done = []
        assert run_units(square, list(range(10)), workers=workers, on_done=done.append) == [x * x for x in range(10)]
        assert sum(done) == 10


def test_pool_uses_several_processes():
    assert len(set(run_units(pid, list(range(12)), workers=3))) > 1


def test_worker_error_propagates():
    with pytest.raises(ValueError, match="unit 3 failed"):
        run_units(boom, list(range(8)), workers=2)


def test_cancel_stops_and_raises():
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(ExportCancelled):
        run_units(square, list(range(200)), workers=2, cancel=cancel)


def test_default_workers():
    assert default_workers() == max(1, (os.cpu_count() or 1) - 4)
