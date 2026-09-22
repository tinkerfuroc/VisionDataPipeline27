"""GPU smoke tests. Skipped without a bf16-capable CUDA device + the converted checkpoint.

The scene is synthetic (a bright disc moving over a noise background) so ground truth is exact.
"""

import contextlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from vsp.engine.device import cuda_capabilities, pick_device
from vsp.ingest import FrameDirSource, import_clip
from vsp.store import Keyframe, LabelObject, ProjectStore, Prompt, now_iso
from vsp.weights import verify_checkpoint

MODEL_DIR = Path(__file__).resolve().parents[2] / "vision_train" / "sam3_checkpoint_hf"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(bool(verify_checkpoint(MODEL_DIR)), reason="converted SAM3 checkpoint not available"),
    pytest.mark.skipif(pick_device("auto", cuda_capabilities()) is None, reason="no bf16-capable CUDA device"),
]

H, W, R = 480, 640, 45


def _disc(cx, cy):
    yy, xx = np.mgrid[:H, :W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= R * R


def _frame(cx, cy, rng):
    img = rng.integers(60, 120, size=(H, W, 3), dtype=np.uint8)
    img[_disc(cx, cy)] = (40, 60, 230)  # BGR: a red disc
    return img


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / max(1, (a | b).sum())


@pytest.fixture(scope="module")
def engine():
    from vsp.engine.sam3 import Sam3Engine

    eng = Sam3Engine(MODEL_DIR)
    eng.load()
    yield eng
    eng.release()


@pytest.fixture(scope="module")
def video():
    rng = np.random.default_rng(0)
    centers = [(120 + 6 * i, 200 + 2 * i) for i in range(20)]
    return [_frame(cx, cy, rng) for cx, cy in centers], [_disc(cx, cy) for cx, cy in centers]


def test_load_selects_bf16_device(engine):
    assert engine.is_loaded()
    assert engine.device is not None and engine.device.startswith("cuda:")


def test_box_prompt_segments_the_disc(engine, video):
    frames, truth = video
    cx, cy = 120, 200
    det = engine.predict_prompt(frames[0], box=(cx - R - 5, cy - R - 5, cx + R + 5, cy + R + 5), frame_key=("t", 0))
    assert det is not None
    assert _iou(det.mask, truth[0]) > 0.85


def test_single_positive_click_segments_the_disc(engine, video):
    frames, truth = video
    det = engine.predict_prompt(frames[0], points=[(120, 200, 1)], frame_key=("t", 0))
    assert det is not None
    assert det.mask.shape == (H, W) and det.mask.dtype == bool
    assert _iou(det.mask, truth[0]) > 0.85


def test_negative_click_is_accepted_together_with_box(engine, video):
    frames, truth = video
    det = engine.predict_prompt(
        frames[0], box=(60, 140, 180, 260), points=[(120, 200, 1), (20, 20, 0)], frame_key=("t", 0)
    )
    assert det is not None and _iou(det.mask, truth[0]) > 0.85


def test_prompt_masks_pass_through_the_degenerate_filter(engine, video, monkeypatch):
    frames, _ = video
    # The disc is ~6362 px of 307200 (2.1 %). Force each bound in turn so the real mask becomes "degenerate".
    monkeypatch.setattr(engine, "MAX_MASK_AREA_FRAC", 0.001)
    assert engine.predict_prompt(frames[0], points=[(120, 200, 1)], frame_key=("t", 0)) is None
    monkeypatch.setattr(engine, "MAX_MASK_AREA_FRAC", 0.45)
    monkeypatch.setattr(engine, "MIN_MASK_AREA_PX", 10**6)
    assert engine.predict_prompt(frames[0], points=[(120, 200, 1)], frame_key=("t", 0)) is None
    monkeypatch.undo()
    assert engine.predict_prompt(frames[0], points=[(120, 200, 1)], frame_key=("t", 0)) is not None


def test_text_prompt_runs_and_returns_ranked_detections(engine, video):
    frames, _ = video
    dets = engine.predict_text(frames[0], "red circle", frame_key=("t", 0))
    assert isinstance(dets, list)  # synthetic art may or may not match; the call must work
    assert [d.score for d in dets] == sorted((d.score for d in dets), reverse=True)
    print(f"\n[text] 'red circle' -> {len(dets)} detection(s)")


def _track(engine, frames, truth):
    got = dict(engine.propagate(frames, truth[0]))
    assert sorted(got) == list(range(len(frames)))
    good = [i for i in range(1, len(frames)) if got[i] is not None and _iou(got[i], truth[i]) > 0.7]
    return len(good)


def test_forward_propagation_tracks_the_disc(engine, video):
    frames, truth = video
    assert _track(engine, frames, truth) >= 18  # of 19 non-seed frames


def test_backward_propagation_is_just_a_reversed_list(engine, video):
    frames, truth = video
    assert _track(engine, frames[::-1], truth[::-1]) >= 18


def test_propagation_streams_masks_instead_of_encoding_the_whole_chunk_first(engine, video, monkeypatch):
    """Frame k's mask must reach the caller before frame k+1 is encoded, so the GUI sees a steady
    stream of frames rather than a long silence followed by a burst of the whole chunk."""
    frames, truth = video
    order: list[str] = []
    real = engine._cache_tracker_features

    def spy(session, frame_idx, embeds):
        order.append(f"encode {frame_idx}")
        real(session, frame_idx, embeds)

    monkeypatch.setattr(engine, "_cache_tracker_features", spy)
    with contextlib.closing(engine.propagate(frames, truth[0])) as masks:
        for offset, _ in masks:
            order.append(f"yield {offset}")
    n = len(frames)
    expected = ["encode 0"] + [s for k in range(n) for s in (f"yield {k}", f"encode {k + 1}")][:-1]
    assert order == expected


def test_end_to_end_store_ingest_engine_store(engine, video, tmp_path):
    """store -> ingest -> real engine -> store, run together for the first time."""
    frames, truth = video
    src = tmp_path / "src"
    src.mkdir()
    for i, fr in enumerate(frames):
        assert cv2.imwrite(str(src / f"frame_{i}.png"), fr)

    store = ProjectStore(tmp_path / "data")
    project = store.create_project("e2e")
    cls = store.add_class(project.id, "disc")
    clip = import_clip(store, project.id, FrameDirSource(src))
    assert clip.frame_count == 20

    img0 = cv2.imread(str(store.frame_path(clip.id, 0)))  # JPEG re-encode of the synthetic frame
    det = engine.predict_prompt(img0, points=[(120, 200, 1)], frame_key=(clip.id, 0))
    assert det is not None
    click_iou = _iou(det.mask, truth[0])
    assert click_iou > 0.85
    store.write_mask(clip.id, 1, 0, det.mask)

    disk_frames = [cv2.imread(str(store.frame_path(clip.id, i))) for i in range(clip.frame_count)]
    with contextlib.closing(engine.propagate(disk_frames, store.read_mask(clip.id, 1, 0))) as gen:
        for offset, mask in gen:
            if offset >= 1 and mask is not None:
                store.write_mask(clip.id, 1, offset, mask)

    n_masks = len(store.mask_indices(clip.id, 1))
    assert n_masks >= 19
    mask19 = store.read_mask(clip.id, 1, 19)
    assert mask19 is not None
    frame19_iou = _iou(mask19, truth[19])
    assert frame19_iou > 0.7
    print(f"\n[e2e] click IoU (JPEG re-encoded frame0): {click_iou:.3f}")
    print(f"[e2e] masks stored for object 1: {n_masks}")
    print(f"[e2e] frame19 IoU: {frame19_iou:.3f}")

    obj = LabelObject(
        id=1,
        class_id=cls.id,
        keyframes=[
            Keyframe(
                frame_idx=0,
                prompt=Prompt(type="clicks", payload={"points": [[120, 200, 1]]}),
                created_at=now_iso(),
            )
        ],
    )
    clip.objects.append(obj)
    store.write_clip(clip)
    reread = store.read_clip(clip.id)
    states = store.frame_states(reread)
    assert states[0] == "keyframe"
    assert states.count("masked") >= 18


def test_peak_vram_fits_the_12gb_card(engine, video):
    import torch

    frames, truth = video
    torch.cuda.reset_peak_memory_stats(engine.device)
    list(engine.propagate(frames + frames[::-1][:4], truth[0]))  # 24 frames = default chunk_size
    peak = torch.cuda.max_memory_allocated(engine.device) / 2**30
    print(f"\n[vram] peak during 24-frame propagate: {peak:.2f} GiB")
    assert peak < 11.0


def test_many_click_refinement_with_a_box_stays_accurate(engine, video):
    """Plan 1 verified box + 2 clicks only; the labeling layer sends a box-from-mask plus every click."""
    frames, truth = video
    cx, cy = 120, 200
    box = (cx - R - 10, cy - R - 10, cx + R + 10, cy + R + 10)
    points = [
        (cx, cy, 1), (cx - 20, cy, 1), (cx + 20, cy - 10, 1),
        (cx - R - 25, cy, 0), (cx + R + 25, cy, 0), (cx, cy + R + 25, 0),
    ]
    det = engine.predict_prompt(frames[0], box=box, points=points, frame_key=("t", 0))
    assert det is not None
    assert _iou(det.mask, truth[0]) > 0.85


def test_many_clicks_without_a_box_stay_accurate(engine, video):
    """The fresh-prompt path: several positive and negative clicks, no box at all."""
    frames, truth = video
    cx, cy = 120, 200
    points = [
        (cx, cy, 1), (cx - 20, cy, 1), (cx + 20, cy - 10, 1),
        (cx - R - 25, cy, 0), (cx + R + 25, cy, 0), (cx, cy + R + 25, 0),
    ]
    det = engine.predict_prompt(frames[0], points=points, frame_key=("t", 0))
    assert det is not None
    assert _iou(det.mask, truth[0]) > 0.85
