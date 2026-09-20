"""SAM3 engine on Hugging Face transformers.

Ported from tinkerfuroc/tk25_vision_train (MIT, (c) 2025 tinker@furoc),
server/tk_vision/annotate/sam3.py @ 5874ba0. Changes from the template:
  * device auto-selection; torch.autocast always gets device_type "cuda";
  * tracker_neck weights are synced from whichever key layout the checkpoint uses;
  * single-object API matching vsp.engine.types.Engine;
  * predict_prompt(): real point/box prompts through the tracker's prompt encoder
    (the template collapsed clicks into a padded box on the detector).
"""

from __future__ import annotations

import collections
import logging
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import torch

from .device import cuda_capabilities, pick_device
from .masks import bbox_of, is_degenerate_mask
from .types import ChunkOOMError, Detection, EngineNotLoaded, Point

log = logging.getLogger(__name__)

_FRAME_CACHE_CAPACITY = 8
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
_OBJ_ID = 1


class Sam3Engine:
    MASK_THRESHOLD = 0.5  # on sigmoid(mask logits); SAM3 masks smear on low-confidence detections
    MAX_DETECTIONS = 8
    MAX_MASK_AREA_FRAC = 0.45
    MIN_MASK_AREA_PX = 64

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        text_threshold: float = 0.45,
        score_mode: str = "native",
    ) -> None:
        self.model_dir = str(Path(model_dir).resolve())
        self.requested_device = device
        self.torch_dtype = _DTYPES[dtype]
        self.text_threshold = float(text_threshold)
        self.score_mode = score_mode
        self.device: str | None = None
        self.load_error: str | None = None
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._tracker_processor = None
        self._cache: collections.OrderedDict[tuple, tuple[torch.Tensor, object]] = collections.OrderedDict()

    # ---------------------------------------------------------------- lifecycle
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                self._load_locked()
                self.load_error = None
            except Exception as e:
                self.load_error = str(e)
                raise

    def _load_locked(self) -> None:
        device = pick_device(self.requested_device, cuda_capabilities())
        if device is None:
            raise EngineNotLoaded("no CUDA device with compute capability >= 8.0 (needed for bfloat16)")
        from transformers import Sam3Processor, Sam3TrackerVideoProcessor, Sam3VideoModel

        self._processor = Sam3Processor.from_pretrained(self.model_dir)
        self._tracker_processor = Sam3TrackerVideoProcessor.from_pretrained(self.model_dir)
        model = Sam3VideoModel.from_pretrained(self.model_dir, dtype=self.torch_dtype)
        synced, expected = self._sync_tracker_neck(model)
        if synced != expected:
            raise EngineNotLoaded(
                f"tracker_neck: synced {synced}/{expected} weights from {self.model_dir}/model.safetensors; "
                "run `vsp verify-weights`"
            )
        # The checkpoint ships tracker_config.multimask_max_pt_num=1, so the mask head only
        # runs its "best of 3 candidates" selection (multimask_output_in_sam) when there is at
        # most one point total. A box prompt already counts as 2 points (its corners), so any
        # box+click combination silently falls back to a single mask token gated only by a
        # stability heuristic (dynamic_multimask_via_stability) -- fine for a box alone or a
        # tight box, but it produces a visibly over-inclusive mask once real click points are
        # combined with a looser box (measured: IoU 0.75 vs >0.9 with this raised). Raising the
        # cap re-enables true multimask selection for realistic prompt combinations (a box plus
        # a handful of refinement clicks); it is a model hyperparameter on the loaded instance,
        # not a change to library code.
        model.tracker_model.config.multimask_max_pt_num = 32
        self._model = model.to(device).eval()
        self.device = device
        log.info("Sam3Engine loaded on %s; vram=%.2f GiB", device, torch.cuda.memory_allocated(device) / 2**30)

    @staticmethod
    def _sync_tracker_neck(model) -> tuple[int, int]:
        """Copy tracker_neck weights straight from the safetensors file.

        HF's loader has been observed to drop these (suffix collision with the detector's neck),
        leaving the tracker FPN at random init. Copying is idempotent when they did load.
        Accepts both key layouts: top-level (official converter) and tracker_model.* (template).
        """
        from safetensors import safe_open

        params = dict(model.tracker_neck.named_parameters())
        synced = 0
        with safe_open(str(Path(model.config.name_or_path) / "model.safetensors"), framework="pt", device="cpu") as f:
            available = set(f.keys())
            for name, param in params.items():
                for key in (f"tracker_neck.{name}", f"tracker_model.tracker_neck.{name}"):
                    if key in available:
                        tensor = f.get_tensor(key)
                        if tensor.shape == param.shape:
                            with torch.no_grad():
                                param.copy_(tensor.to(param.dtype))
                            synced += 1
                        break
        return synced, len(params)

    def release(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            self._tracker_processor = None
            self._cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _require_loaded(self) -> None:
        if self._model is None:
            raise EngineNotLoaded(self.load_error or "Sam3Engine.load() has not been called")

    def _autocast(self):
        return torch.autocast("cuda", dtype=self.torch_dtype)

    # ------------------------------------------------------------ vision cache
    def _vision_locked(self, image_bgr: np.ndarray, frame_key: tuple | None):
        """Returns (pixel_values [1,3,1008,1008], detector vision embeds). Caller holds the lock."""
        if frame_key is not None and frame_key in self._cache:
            self._cache.move_to_end(frame_key)
            return self._cache[frame_key]
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        pixel_values = self._processor(images=rgb, return_tensors="pt")["pixel_values"].to(self.device)
        embeds = self._model.detector_model.get_vision_features(pixel_values=pixel_values)
        if frame_key is not None:
            self._cache[frame_key] = (pixel_values, embeds)
            while len(self._cache) > _FRAME_CACHE_CAPACITY:
                self._cache.popitem(last=False)
        return pixel_values, embeds

    # ------------------------------------------------------------------- text
    def predict_text(self, image_bgr: np.ndarray, phrase: str, *, frame_key: tuple | None = None) -> list[Detection]:
        self._require_loaded()
        h, w = image_bgr.shape[:2]
        with self._lock, torch.inference_mode(), self._autocast():
            _, embeds = self._vision_locked(image_bgr, frame_key)
            enc = self._processor.tokenizer(
                phrase, return_tensors="pt", padding="max_length", max_length=32, truncation=True
            )
            out = self._model.detector_model(
                vision_embeds=embeds,
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )
            scores, masks = self._decode(out, (h, w))
        dets: list[Detection] = []
        for idx in np.argsort(-scores):
            if len(dets) >= self.MAX_DETECTIONS:
                break
            score, mask = float(scores[idx]), masks[idx]
            if self.score_mode == "per_query" and score <= self.text_threshold:
                continue
            if is_degenerate_mask(mask, max_area_frac=self.MAX_MASK_AREA_FRAC, min_area_px=self.MIN_MASK_AREA_PX):
                continue
            dets.append(Detection(mask=mask, bbox=bbox_of(mask), score=score, phrase=phrase))
        return dets

    def _decode(self, out, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        h, w = size
        if self.score_mode == "per_query":  # ignore the presence head (verbose prompts)
            scores = out.pred_logits[0].detach().float().sigmoid()
            low = out.pred_masks[0].detach().float().sigmoid()
            full = torch.nn.functional.interpolate(low.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
            return scores.cpu().numpy(), (full.squeeze(0) > self.MASK_THRESHOLD).cpu().numpy()
        res = self._processor.post_process_instance_segmentation(
            out, threshold=self.text_threshold, mask_threshold=self.MASK_THRESHOLD, target_sizes=[(h, w)]
        )
        if not res or res[0].get("scores") is None or len(res[0]["scores"]) == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0, h, w), dtype=bool)
        r = res[0]
        return r["scores"].detach().float().cpu().numpy(), r["masks"].detach().cpu().numpy().astype(bool)

    # ---------------------------------------------------------- points / box
    def _new_session(self, video: torch.Tensor, h: int, w: int):
        from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
            Sam3TrackerVideoInferenceSession,
        )

        n = int(video.shape[0])
        session = Sam3TrackerVideoInferenceSession(
            video=video, video_height=h, video_width=w,
            inference_device=self.device, inference_state_device="cpu", video_storage_device=self.device,
            dtype=self.torch_dtype,
            # The tracker inside Sam3VideoModel has no vision encoder; every frame must be pre-cached.
            max_vision_features_cache_size=max(n, 1),
        )
        return session

    def _cache_tracker_features(self, session, frame_idx: int, vision_embeds) -> None:
        feats, pos = self._model.get_vision_features_for_tracker(vision_embeds=vision_embeds)
        session.cache.cache_vision_features(frame_idx, {"vision_feats": feats, "vision_pos_embeds": pos})

    def _binarize(self, pred_masks: torch.Tensor, h: int, w: int) -> np.ndarray:
        full = torch.nn.functional.interpolate(pred_masks.float(), size=(h, w), mode="bilinear", align_corners=False)
        return (full.sigmoid() > self.MASK_THRESHOLD)[0, 0].cpu().numpy()

    def predict_prompt(
        self,
        image_bgr: np.ndarray,
        *,
        points: Sequence[Point] = (),
        box: tuple[int, int, int, int] | None = None,
        frame_key: tuple | None = None,
    ) -> Detection | None:
        if not points and box is None:
            return None
        self._require_loaded()
        h, w = image_bgr.shape[:2]
        with self._lock, torch.inference_mode(), self._autocast():
            pixel_values, embeds = self._vision_locked(image_bgr, frame_key)
            session = self._new_session(pixel_values, h, w)
            self._cache_tracker_features(session, 0, embeds)
            kwargs = {}
            if points:
                kwargs["input_points"] = [[[[float(x), float(y)] for x, y, _ in points]]]
                kwargs["input_labels"] = [[[int(lbl) for _, _, lbl in points]]]
            if box is not None:
                kwargs["input_boxes"] = [[[float(v) for v in box]]]
            self._tracker_processor.add_inputs_to_inference_session(
                inference_session=session, frame_idx=0, obj_ids=_OBJ_ID, **kwargs
            )
            out = self._model.tracker_model(inference_session=session, frame_idx=0)
            mask = self._binarize(out.pred_masks, h, w)
            score = float(out.object_score_logits.float().sigmoid().flatten()[0])
        if is_degenerate_mask(mask, max_area_frac=self.MAX_MASK_AREA_FRAC, min_area_px=self.MIN_MASK_AREA_PX):
            return None
        return Detection(mask=mask, bbox=bbox_of(mask), score=score)

    # ------------------------------------------------------------- propagate
    def propagate(
        self,
        frames: list[np.ndarray],
        seed_mask: np.ndarray,
        cancel_cb: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[int, np.ndarray | None]]:
        self._require_loaded()
        if not frames:
            return
        h, w = frames[0].shape[:2]
        if seed_mask.shape != (h, w):
            raise ValueError(f"seed_mask shape {seed_mask.shape} does not match frames {(h, w)}")
        if not seed_mask.any():
            raise ValueError("seed_mask must be non-empty")
        rgb = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames]
        video = self._processor(images=rgb, return_tensors="pt")["pixel_values"]
        try:
            with self._lock, torch.inference_mode(), self._autocast():
                yield from self._propagate_locked(video, seed_mask, h, w, cancel_cb)
        except torch.cuda.OutOfMemoryError as e:
            raise ChunkOOMError(str(e)) from e
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise ChunkOOMError(str(e)) from e
            raise
        finally:
            torch.cuda.empty_cache()

    def _propagate_locked(self, video, seed_mask, h, w, cancel_cb):
        n = int(video.shape[0])
        session = self._new_session(video, h, w)
        for fr in range(n):
            px = session.get_frame(fr).unsqueeze(0)
            embeds = self._model.detector_model.get_vision_features(pixel_values=px)
            self._cache_tracker_features(session, fr, embeds)
        obj_idx = session.obj_id_to_idx(_OBJ_ID)
        seed = torch.from_numpy(seed_mask.astype(np.float32))[None, None].to(self.device, dtype=self.torch_dtype)
        session.add_mask_inputs(obj_idx, 0, seed)
        session.obj_with_new_inputs = [_OBJ_ID]
        for out in self._model.tracker_model.propagate_in_video_iterator(
            inference_session=session, start_frame_idx=0, max_frame_num_to_track=n
        ):
            if cancel_cb is not None and cancel_cb():
                return
            mask = None
            if out.pred_masks is not None and len(out.pred_masks) > 0:
                m = self._binarize(out.pred_masks, h, w)
                if not is_degenerate_mask(m, max_area_frac=self.MAX_MASK_AREA_FRAC, min_area_px=self.MIN_MASK_AREA_PX):
                    mask = m
            yield int(out.frame_idx), mask
