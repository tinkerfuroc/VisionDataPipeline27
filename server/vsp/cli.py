"""`vsp` command line. Subcommands are added by later tasks/plans."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, Settings
from .engine.device import cuda_capabilities, pick_device
from .weights import convert_checkpoint, verify_checkpoint


def _load_settings(config: str | None) -> tuple[Settings, Path | None]:
    """Returns the loaded settings and the absolute config path used (None for built-in defaults)."""
    if config:
        path = Path(config).resolve()
        return Settings.load(config), path
    if DEFAULT_CONFIG_PATH.is_file():
        return Settings.load(DEFAULT_CONFIG_PATH), DEFAULT_CONFIG_PATH
    return Settings.load(None), None


def _report(problems: list[str], target: str) -> int:
    if not problems:
        print(f"OK: {target}")
        return 0
    for p in problems:
        print(f"PROBLEM: {p}")
    return 1


def _doctor(settings: Settings, config_path: Path | None) -> int:
    import torch

    print(f"config:    {config_path if config_path is not None else '(built-in defaults)'}")
    print(f"data_root: {settings.project.data_root}")
    print(f"model_dir: {settings.sam3.model_dir}")
    print(f"torch {torch.__version__} (CUDA build {torch.version.cuda})")
    caps = cuda_capabilities()
    for i, cap in enumerate(caps):
        gib = torch.cuda.get_device_properties(i).total_memory / 2**30
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)} cc={cap[0]}.{cap[1]} {gib:.1f} GiB")
    if not caps:
        print("FAIL: CUDA is not available to this torch build")
        return 1
    dev = pick_device(settings.sam3.device, caps)
    if dev is None:
        print("FAIL: no CUDA device with compute capability >= 8.0 (bfloat16)")
        return 1
    print(f"selected: {dev}")
    a = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
    ok = bool(torch.isfinite((a @ a).float()).all())
    q = torch.randn(1, 8, 256, 64, device=dev, dtype=torch.bfloat16)
    torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize(dev)
    print(f"bfloat16 matmul + attention: {'ok' if ok else 'FAIL'}")
    return 0 if ok else 1


def _serve(settings: Settings, host: str | None, port: int | None) -> int:
    import uvicorn

    from .api.app import create_app

    host = host or settings.server.host
    port = port or settings.server.port
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: binding {host} exposes an unauthenticated labeling server to the network")
    print(f"data_root: {settings.project.data_root}")
    print(f"model_dir: {settings.sam3.model_dir}")
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")
    return 0


def _fmt(v: float | None) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def _train(settings: Settings, args, *, gpus=None, worker_cmd=None, poll: float = 2.0) -> int:
    import signal
    import threading
    import time

    from pydantic import ValidationError

    from .labeling.errors import BadRequest, Conflict
    from .store import NotFound
    from .train.models import DEFAULT_MODELS, TrainParams, find_export, list_exports
    from .train.results import BOX_MAP, MASK_MAP, MASK_MAP50, read_metrics
    from .train.scheduler import WORKER_CMD, TrainScheduler, default_gpu, detect_gpus

    data_root = settings.project.data_root
    if args.list:
        for e in list_exports(data_root):
            print(f"{e.id}  {e.project_name}  train {e.train_images}  val {e.val_images}  "
                  f"classes: {', '.join(e.classes)}")
        return 0
    if not args.export:
        print("error: --export is required (see `vsp train --list`)")
        return 2
    try:
        params = TrainParams(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience,
                             workers=args.workers, seed=args.seed)
    except ValidationError as e:
        print(f"error: {e}")
        return 2
    gpus = detect_gpus() if gpus is None else gpus
    if not gpus:
        print("FAIL: no CUDA GPU is available to train on")
        return 1
    if args.gpu is not None:
        gpu = args.gpu
    else:
        try:
            sam3 = pick_device(settings.sam3.device, cuda_capabilities())
        except ValueError:
            sam3 = None
        gpu = default_gpu(gpus, int(sam3.split(":")[1]) if sam3 else None)
    try:
        export = find_export(data_root, args.export)
        sched = TrainScheduler(data_root, gpus, worker_cmd=worker_cmd or WORKER_CMD)
        ids = sched.submit(export, [(m, gpu) for m in (args.model or DEFAULT_MODELS)], params)
    except (NotFound, BadRequest) as e:
        print(f"error: {e}")
        return 2
    print(f"training {len(ids)} model(s) on cuda:{gpu} from export {export.id} ({export.project_name}); "
          f"runs in {sched.runs.root}")
    shown = {rid: 0 for rid in ids}
    states = {rid: "queued" for rid in ids}

    def report() -> None:
        for rid in ids:
            run = sched.runs.read(rid)
            rows = read_metrics(sched.runs.dir(rid))
            for row in rows[shown[rid]:]:
                print(f"{run['model']:<12} epoch {row['epoch']:>3}/{run['params']['epochs']}  "
                      f"mask mAP50-95 {_fmt(row.get(MASK_MAP))}  mask mAP50 {_fmt(row.get(MASK_MAP50))}  "
                      f"box mAP50-95 {_fmt(row.get(BOX_MAP))}", flush=True)
            shown[rid] = len(rows)
            if run["status"] != states[rid]:
                states[rid] = run["status"]
                extra = f": {run['error']}" if run["status"] == "error" else ""
                print(f"{run['model']:<12} {run['status']}{extra}", flush=True)

    def _raise_interrupt(signum, frame) -> None:
        raise KeyboardInterrupt

    # Closing the terminal or `kill`ing this process sends SIGHUP/SIGTERM, not SIGINT; route them
    # through the same cancel path as Ctrl-C so the workers are not left running unowned. Only the
    # main thread may install signal handlers.
    installed = threading.current_thread() is threading.main_thread()
    previous_handlers = {}
    if installed:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            previous_handlers[sig] = signal.signal(sig, _raise_interrupt)

    sched.start(recover=False)
    try:
        while not sched.idle():
            report()
            time.sleep(poll)
    except KeyboardInterrupt:
        print("stopping: cancelling this command's runs…", flush=True)
        for rid in ids:
            if sched.runs.read(rid)["status"] in ("queued", "running"):
                with contextlib.suppress(Conflict):
                    sched.cancel(rid)
        while not sched.idle():
            time.sleep(0.2)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        sched.shutdown()
    report()
    print(f"\n{'model':<12} {'status':<11} {'best epoch':>10} {'mask mAP50':>10} {'mask mAP50-95':>13} "
          f"{'box mAP50-95':>12}  run")
    ok = True
    for rid in ids:
        run = sched.runs.read(rid)
        b = run.get("best") or {}
        ok = ok and run["status"] == "done"
        print(f"{run['model']:<12} {run['status']:<11} {b.get('epoch', '-')!s:>10} "
              f"{_fmt(b.get('mask_map50')):>10} {_fmt(b.get('mask_map50_95')):>13} "
              f"{_fmt(b.get('box_map50_95')):>12}  {rid}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vsp")
    parser.add_argument("--config", default=None, help="path to a YAML config (default: configs/default.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check the torch build, GPU selection and bfloat16 support")
    sub.add_parser("verify-weights", help="check that sam3.model_dir is a complete converted checkpoint")
    p_conv = sub.add_parser("convert-weights", help="convert Meta's public sam3.pt to HF format")
    p_conv.add_argument("--checkpoint", required=True, help="path to sam3.pt")
    p_conv.add_argument(
        "--converter", required=True,
        help="path to convert_sam3_video_to_hf.py downloaded from the transformers GitHub repo",
    )
    p_conv.add_argument("--out", required=True, help="output directory (HF format)")
    p_conv.add_argument(
        "--official-files-from", required=True,
        help="directory holding the official tokenizer/processor files (the facebook/sam3 repo files)",
    )
    p_serve = sub.add_parser("serve", help="run the HTTP API (and the web UI when web/dist exists)")
    p_serve.add_argument("--host", default=None, help="bind address (default: server.host from the config)")
    p_serve.add_argument("--port", type=int, default=None, help="port (default: server.port from the config)")
    p_train = sub.add_parser("train", help="train YOLO segmentation models on a yolo_seg export")
    p_train.add_argument("--list", action="store_true", help="list the exports that have a yolo_seg dataset")
    p_train.add_argument("--export", default=None, help="export id (a folder name under data/exports/)")
    p_train.add_argument("--model", action="append", default=None,
                         help="model to train, repeatable (default: yolo26s/m-seg and yolo11s/m-seg)")
    p_train.add_argument("--gpu", type=int, default=None,
                         help="CUDA index (default: the first GPU SAM3 is not on)")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--imgsz", type=int, default=960)
    p_train.add_argument("--batch", type=int, default=8)
    p_train.add_argument("--patience", type=int, default=30)
    p_train.add_argument("--workers", type=int, default=8)
    p_train.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    settings, config_path = _load_settings(args.config)
    if args.cmd == "doctor":
        return _doctor(settings, config_path)
    if args.cmd == "verify-weights":
        return _report(verify_checkpoint(settings.sam3.model_dir), str(settings.sam3.model_dir))
    if args.cmd == "convert-weights":
        problems = convert_checkpoint(
            Path(args.checkpoint), Path(args.converter), Path(args.out), Path(args.official_files_from)
        )
        return _report(problems, args.out)
    if args.cmd == "serve":
        return _serve(settings, args.host, args.port)
    if args.cmd == "train":
        return _train(settings, args)
    parser.error(f"unknown command {args.cmd}")
    return 2
