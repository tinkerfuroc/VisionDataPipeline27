import multiprocessing
import os
import time

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


def slow(x):
    time.sleep(0.05)
    return x


def test_pool_cancel_stops_and_raises():
    start = time.monotonic()
    with pytest.raises(ExportCancelled):
        run_units(slow, list(range(400)), workers=2, cancel=lambda: True)
    assert time.monotonic() - start < 5  # 400 units x 0.05 s on 2 workers would take >= 10 s
    assert multiprocessing.active_children() == []


def test_inline_cancel_stops_before_further_units():
    done = []
    with pytest.raises(ExportCancelled):
        run_units(square, list(range(10)), workers=1, on_done=done.append, cancel=lambda: len(done) >= 3)
    assert sum(done) == 3


def test_default_workers():
    assert default_workers() == max(1, (os.cpu_count() or 1) - 4)
