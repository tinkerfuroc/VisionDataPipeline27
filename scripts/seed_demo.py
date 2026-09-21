"""Create a demo project with a synthetic moving ball through the HTTP API (manual QA aid for the web GUI).

The scene is deliberately colourful: a six-colour ball spins along a wavy path over a hue-gradient background and
passes behind-the-colour of several distractors (one of them close to the default class colour), so the mask overlay
and its outline can be judged against many hues rather than one flat grey.

Usage (server running):  uv run python scripts/seed_demo.py [--base http://127.0.0.1:28000] [--frames 40] [--no-label]

The synthetic frames are written to data/demo_frames under the repository root — inside the server's default
allowed root ("~") — and are overwritten in place on every run. Each run creates another "demo" project (the
server does not dedupe by name), so re-running this script repeatedly accumulates projects.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = REPO_ROOT / "data" / "demo_frames"  # under the repository, hence inside the default allowed root "~"
W, H, R = 640, 480, 45
WAIT_TIMEOUT_S = 600.0


WEDGES_BGR = [(60, 60, 230), (40, 140, 245), (50, 220, 240), (90, 190, 70), (220, 130, 50), (170, 70, 150)]


def ball_center(i: int) -> tuple[int, int]:
    """Left to right along a sine wave, so the ball crosses the whole hue gradient and most distractors."""
    return 110 + 11 * i, int(240 + 95 * math.sin(i / 6.0))


def background() -> np.ndarray:
    xs = np.linspace(0, 179, W, dtype=np.float32)[None, :].repeat(H, 0)  # hue sweeps left to right
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None].repeat(W, 1)
    hsv = np.dstack([xs, 110 + 60 * ys, 90 + 90 * (1 - ys)]).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rng = np.random.default_rng(7)  # fixed seed: the texture must not flicker between frames
    noise = cv2.GaussianBlur(rng.normal(0, 22, (H, W)).astype(np.float32), (0, 0), 3)[..., None]
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # static distractors; the crimson bar is close to the default class colour (#e6194b) on purpose
    cv2.rectangle(img, (250, 120), (330, 380), (75, 25, 230), -1)
    cv2.rectangle(img, (430, 40), (580, 130), (235, 235, 235), -1)
    cv2.circle(img, (470, 360), 60, (30, 30, 30), -1)
    cv2.ellipse(img, (150, 90), (70, 35), 20, 0, 360, (60, 200, 250), -1)
    return img


def write_frames(n: int) -> None:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAMES_DIR.glob("*.png"):
        old.unlink()
    base = background()
    for i in range(n):
        img = base.copy()
        cx, cy = ball_center(i)
        spin = 9 * i  # degrees
        for k, color in enumerate(WEDGES_BGR):
            cv2.ellipse(img, (cx, cy), (R, R), spin, 60 * k, 60 * (k + 1), color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), R // 4, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), R, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.imwrite(str(FRAMES_DIR / f"{i:04d}.png"), img)


def wait_job(client: httpx.Client, job_id: str, timeout: float = WAIT_TIMEOUT_S) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        r = client.get(f"/api/jobs/{job_id}")
        r.raise_for_status()
        job = r.json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
        if time.monotonic() >= deadline:
            raise TimeoutError(f"job {job_id} did not finish within {timeout:.0f}s (last status: {job['status']!r})")
        time.sleep(0.2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:28000")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--no-label", action="store_true", help="import only; do not accept a prompt or propagate")
    args = ap.parse_args()

    write_frames(args.frames)
    # trust_env=False: this machine exports http_proxy, which must not be used for localhost
    with httpx.Client(base_url=args.base, trust_env=False, timeout=120) as client:
        health_resp = client.get("/api/health")
        health_resp.raise_for_status()

        project_resp = client.post("/api/projects", json={"name": "demo"})
        project_resp.raise_for_status()
        project = project_resp.json()

        class_resp = client.post(f"/api/projects/{project['id']}/classes", json={"name": "ball"})
        class_resp.raise_for_status()
        cls = class_resp.json()

        started = client.post(
            f"/api/projects/{project['id']}/clips/import-frames", json={"path": str(FRAMES_DIR), "fps": 10, "stride": 1}
        )
        started.raise_for_status()
        job = wait_job(client, started.json()["job_id"])
        if job["status"] != "done":
            print("import failed:", job["error"], file=sys.stderr)
            return 1
        clip_id = job["result"]["clip_id"]
        print(f"clip {clip_id}: {job['result']['frame_count']} frames")

        if not args.no_label:
            # Re-read health rather than reuse the pre-import snapshot: the engine may have finished loading
            # in the background while the import (and its wait loop) ran.
            health_resp = client.get("/api/health")
            health_resp.raise_for_status()
            health = health_resp.json()["engine"]
            if not health["loaded"]:
                print("engine not loaded (yet) — imported only:", health["error"] or "still loading", file=sys.stderr)
            else:
                # a box, not a click: a click on one wedge tends to select that wedge alone
                cx, cy = ball_center(0)
                prompt = {"type": "box", "box": [cx - R - 4, cy - R - 4, cx + R + 4, cy + R + 4]}
                client.post(f"/api/clips/{clip_id}/accept", json={"frame": 0, "prompt": prompt, "class_id": cls["id"]}).raise_for_status()
                started = client.post(f"/api/clips/{clip_id}/propagate", json={"from_frame": 0, "direction": "forward"})
                started.raise_for_status()
                result = wait_job(client, started.json()["job_id"])
                print("propagate:", result["status"], result["result"])
        print(f"open {args.base}/#/clips/{clip_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
