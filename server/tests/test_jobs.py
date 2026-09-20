import threading

import pytest
from vsp.jobs import INTERACTIVE, TERMINAL, JobBusy, JobManager, JobNotFound


def test_job_runs_and_finishes_with_result_and_events():
    jobs = JobManager()

    def work(job):
        job.emit({"type": "progress", "done": 1})
        return {"clip_id": "abc"}

    job = jobs.start("import", work)
    job.join(5)
    assert job.status == "done" and job.result == {"clip_id": "abc"} and job.error is None
    events = job.events_since(0)
    assert [e["type"] for e in events] == ["progress", "done"]
    assert [e["seq"] for e in events] == [0, 1]
    assert events[-1]["clip_id"] == "abc"
    assert job.events_since(1) == events[1:]
    snap = job.snapshot()
    assert snap["status"] == "done" and snap["n_events"] == 2 and snap["kind"] == "import"
    assert jobs.get(job.id) is job


def test_exception_becomes_error_status():
    jobs = JobManager()

    def boom(job):
        raise ValueError("bad video")

    job = jobs.start("import", boom)
    job.join(5)
    assert job.status == "error" and job.error == "ValueError: bad video"
    assert job.events_since(0)[-1] == {"type": "error", "message": "ValueError: bad video", "seq": 0}


def test_cancel_is_cooperative():
    jobs = JobManager()
    started = threading.Event()

    def work(job):
        started.set()
        while not job.cancel_requested():
            threading.Event().wait(0.01)
        return {"processed": 3, "cancelled": True}

    job = jobs.start("propagate", work, exclusive="gpu")
    assert started.wait(5)
    assert jobs.running("gpu") == job.id
    jobs.cancel(job.id)
    job.join(5)
    assert job.status == "cancelled" and job.result == {"processed": 3, "cancelled": True}
    assert job.events_since(0)[-1]["type"] == "cancelled"
    assert jobs.running("gpu") is None


def test_status_reflects_what_the_work_function_reports_not_just_a_cancel_request():
    """A job is 'cancelled' only when its fn's result says it stopped early -- never merely because
    request_cancel() was called. A fn that ignores the request and finishes must end 'done'."""
    jobs = JobManager()
    started = threading.Event()

    def ignores_cancel(job):
        started.set()
        while not job.cancel_requested():
            threading.Event().wait(0.01)
        return {"x": 1}  # finished the real work; cancellation was requested but not honoured

    job = jobs.start("x", ignores_cancel)
    assert started.wait(5)
    jobs.cancel(job.id)
    job.join(5)
    assert job.status == "done"
    assert job.events_since(0)[-1]["type"] == "done"

    honours_cancel = jobs.start("x", lambda job: {"cancelled": True})
    honours_cancel.join(5)
    assert honours_cancel.status == "cancelled"


def test_exclusive_key_allows_one_job_at_a_time():
    jobs = JobManager()
    release = threading.Event()
    first = jobs.start("propagate", lambda job: release.wait(5) and None, exclusive="gpu")
    with pytest.raises(JobBusy) as busy:
        jobs.start("propagate", lambda job: None, exclusive="gpu")
    assert busy.value.job_id == first.id
    other = jobs.start("import", lambda job: None)  # no exclusive key -> not blocked
    other.join(5)
    release.set()
    first.join(5)
    assert first.status in TERMINAL
    jobs.start("propagate", lambda job: None, exclusive="gpu").join(5)  # free again


def test_final_event_is_visible_before_terminal_status():
    jobs = JobManager()
    job = jobs.start("x", lambda job: {"n": 1})
    job.join(5)
    # whenever status is terminal, the last event must already be the terminal one
    assert job.status in TERMINAL and job.events_since(0)[-1]["type"] == job.status


def test_unknown_job():
    with pytest.raises(JobNotFound):
        JobManager().get("nope")


def test_a_lease_blocks_job_start_and_is_released_on_exit():
    jobs = JobManager()
    with jobs.lease("gpu"):
        with pytest.raises(JobBusy) as busy:
            jobs.start("propagate", lambda job: None, exclusive="gpu")
        assert busy.value.job_id == INTERACTIVE
    jobs.start("propagate", lambda job: None, exclusive="gpu").join(5)  # released on exit


def test_a_running_job_blocks_a_lease():
    jobs = JobManager()
    release = threading.Event()
    job = jobs.start("propagate", lambda job: release.wait(5) and None, exclusive="gpu")
    with pytest.raises(JobBusy) as busy, jobs.lease("gpu"):
        pass
    assert busy.value.job_id == job.id
    release.set()
    job.join(5)
    with jobs.lease("gpu"):  # free again once the job is terminal
        pass


def test_leases_do_not_exclude_each_other_and_survive_exceptions():
    jobs = JobManager()
    with jobs.lease("gpu"), jobs.lease("gpu"):  # overlapping leases on the same key both succeed
        jobs.start("import", lambda job: None).join(5)  # no exclusive key -> not blocked
        jobs.start("other", lambda job: None, exclusive="other").join(5)  # different key -> not blocked

    with pytest.raises(ValueError, match="boom"), jobs.lease("gpu"):
        raise ValueError("boom")
    jobs.start("propagate", lambda job: None, exclusive="gpu").join(5)  # the raising lease still released


def test_leases_are_not_jobs():
    jobs = JobManager()
    with jobs.lease("gpu"):
        assert jobs.running("gpu") is None
        with pytest.raises(JobNotFound):
            jobs.get(INTERACTIVE)
