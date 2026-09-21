"""Create a demo project with a synthetic moving disc through the HTTP API (manual QA aid for the web GUI).

Usage (server running):  uv run python scripts/seed_demo.py [--base http://127.0.0.1:28000] [--frames 40] [--no-label]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = REPO_ROOT / "data" / "demo_frames"  # under the repository, hence inside the default allowed root "~"
W, H, R = 640, 480, 45


def write_frames(n: int) -> None:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAMES_DIR.glob("*.png"):
        old.unlink()
    for i in range(n):
        img = np.full((H, W, 3), 40, np.uint8)
        cv2.rectangle(img, (420, 60), (560, 160), (90, 90, 90), -1)  # a static distractor
        cv2.circle(img, (120 + 8 * i, 240), R, (30, 200, 230), -1)
        cv2.imwrite(str(FRAMES_DIR / f"{i:04d}.png"), img)


def wait_job(client: httpx.Client, job_id: str) -> dict:
    while True:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
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
        health = client.get("/api/health").json()["engine"]
        project = client.post("/api/projects", json={"name": "demo"}).json()
        cls = client.post(f"/api/projects/{project['id']}/classes", json={"name": "disc"}).json()
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
            if not health["loaded"]:
                print("engine not loaded (yet) — imported only:", health["error"] or "still loading", file=sys.stderr)
            else:
                prompt = {"type": "clicks", "points": [[120, 240, 1]]}
                client.post(f"/api/clips/{clip_id}/accept", json={"frame": 0, "prompt": prompt, "class_id": cls["id"]}).raise_for_status()
                started = client.post(f"/api/clips/{clip_id}/propagate", json={"from_frame": 0, "direction": "forward"})
                started.raise_for_status()
                result = wait_job(client, started.json()["job_id"])
                print("propagate:", result["status"], result["result"])
        print(f"open {args.base}/#/clips/{clip_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
