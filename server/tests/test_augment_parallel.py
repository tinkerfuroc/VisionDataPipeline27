import multiprocessing
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import cv2
import pytest
from vdp import parallel
from vdp.export import ExportCancelled
from vdp.parallel import default_workers, run_units


def square(x):
    return x * x


def boom(x):
    if x == 3:
        raise ValueError("unit 3 failed")
    return x


def pid(_):
    # long enough that the first worker cannot drain every unit before the others have started (spawn start-up
    # takes about a second on a slow CI runner)
    time.sleep(0.5)
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
    assert default_workers() == max(1, len(os.sched_getaffinity(0)) - 4)


def test_cancel_after_the_last_unit_returns_results():
    done = []
    assert run_units(square, [1, 2], workers=2, cancel=lambda: len(done) >= 2, on_done=done.append) == [1, 4]


def threads_and_niceness(_):
    return cv2.getNumThreads(), os.nice(0)


def test_pool_workers_run_at_low_priority_with_one_opencv_thread():
    for threads, nice in run_units(threads_and_niceness, [0, 1], workers=2):
        assert threads == 1 and nice >= 10


def test_inline_run_leaves_the_calling_process_alone():
    before = cv2.getNumThreads(), os.nice(0)
    run_units(square, [1, 2], workers=1)
    assert (cv2.getNumThreads(), os.nice(0)) == before


def nap(x):
    time.sleep(3)
    return x


def test_pool_checks_cancel_while_units_run():
    # cancel is polled on a timed wake-up, not only when a unit finishes, so its latency is not a unit's length
    calls = []
    assert run_units(nap, [0, 1], workers=2, cancel=lambda: calls.append(1) or False) == [0, 1]
    assert len(calls) >= 3  # two 3 s units finishing together would give at most one check without the timeout


_ORPHAN_HELPER = textwrap.dedent('''
    import os, sys, time
    from pathlib import Path
    from vdp.parallel import run_units

    def unit(i):
        (Path(sys.argv[1]) / f"{os.getpid()}-{os.getppid()}").touch()
        time.sleep(120)
        return i

    if __name__ == "__main__":
        run_units(unit, list(range(4)), workers=2)
''')


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_workers_exit_when_the_parent_is_killed(tmp_path):
    script = tmp_path / "helper.py"
    script.write_text(_ORPHAN_HELPER)
    marks = tmp_path / "marks"
    marks.mkdir()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    helper = subprocess.Popen([sys.executable, str(script), str(marks)], env=env)
    workers: list[int] = []
    try:
        deadline = time.monotonic() + 120
        while len(workers) < 2 and time.monotonic() < deadline and helper.poll() is None:
            time.sleep(0.1)
            workers = [int(m.name.split("-")[0]) for m in marks.iterdir() if m.name.endswith(f"-{helper.pid}")]
        assert len(workers) == 2, "the helper's workers never started"
        os.kill(helper.pid, signal.SIGKILL)
        helper.wait(10)
        deadline = time.monotonic() + 15
        while any(_alive(p) for p in workers) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not any(_alive(p) for p in workers)
    finally:
        if helper.poll() is None:
            helper.kill()
        for p in workers:
            if _alive(p):
                os.kill(p, signal.SIGKILL)


def test_inline_deadline_already_passed_runs_nothing():
    calls = []

    def fn(x):
        calls.append(x)
        return x

    assert run_units(fn, [1, 2, 3], workers=1, deadline=time.monotonic() - 1) == [None, None, None]
    assert calls == []


def test_pool_deadline_keeps_finished_units_and_drops_the_rest():
    done = []
    start = time.monotonic()
    results = run_units(slow, list(range(200)), workers=2, on_done=done.append, deadline=start + 0.5)
    assert time.monotonic() - start < 5  # 200 units x 0.05 s on 2 workers would take >= 5 s plus start-up
    assert len(results) == 200
    kept = [i for i, r in enumerate(results) if r is not None]
    assert 0 < len(kept) < 200
    assert all(results[i] == i for i in kept)
    assert sum(done) == len(kept)
    assert multiprocessing.active_children() == []


def test_cancel_wins_over_the_deadline():
    with pytest.raises(ExportCancelled):
        run_units(slow, list(range(200)), workers=2, cancel=lambda: True, deadline=time.monotonic() - 1)
    assert multiprocessing.active_children() == []


def test_pool_deadline_already_passed_starts_no_pool(monkeypatch):
    from vdp import parallel

    def no_pool(*args, **kwargs):
        raise AssertionError("a pool was started after the deadline")

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", no_pool)
    done = []
    results = run_units(square, [1, 2, 3], workers=2, on_done=done.append, deadline=time.monotonic() - 1)
    assert results == [None, None, None] and done == []


def test_on_result_receives_each_result_and_none_is_kept():
    for workers in (1, 3):
        seen = {}
        results = run_units(square, list(range(6)), workers=workers, on_result=seen.__setitem__)
        assert results == [None] * 6
        assert seen == {i: i * i for i in range(6)}


GIB = 2**30


def test_memory_cap_limits_workers_to_what_fits(monkeypatch):
    monkeypatch.setattr(parallel, "available_memory", lambda: parallel.MEMORY_RESERVE + 3 * parallel.WORKER_MEMORY)
    assert parallel.memory_capped_workers(16) == 3
    assert parallel.memory_capped_workers(2) == 2  # never raises the count


def test_memory_cap_keeps_at_least_one_worker(monkeypatch):
    monkeypatch.setattr(parallel, "available_memory", lambda: 0)
    assert parallel.memory_capped_workers(16) == 1


def test_memory_cap_is_skipped_when_memory_cannot_be_read(monkeypatch):
    monkeypatch.setattr(parallel, "available_memory", lambda: None)
    assert parallel.memory_capped_workers(16) == 16


def test_available_memory_reads_memavailable():
    avail = parallel.available_memory()
    assert avail is None or 0 < avail < 2**50


def test_run_units_applies_the_memory_cap(monkeypatch):
    monkeypatch.setattr(parallel, "available_memory", lambda: 0)  # room for one: runs inline, in this process
    assert set(run_units(pid_now, list(range(4)), workers=4)) == {os.getpid()}


def pid_now(_):
    return os.getpid()


class Blob:
    """A result type that can be weakly referenced, to see whether run_units still holds it."""

    def __init__(self, x):
        self.x = x


def make_blob(x):
    return Blob(x)


def test_on_result_results_are_not_kept_alive():
    import gc
    import weakref

    for workers in (1, 3):
        refs = []

        def on_result(i, blob, refs=refs):
            gc.collect()
            assert all(r() is None for r in refs), "an earlier result is still referenced"
            refs.append(weakref.ref(blob))

        run_units(make_blob, list(range(8)), workers=workers, on_result=on_result)
        assert len(refs) == 8
