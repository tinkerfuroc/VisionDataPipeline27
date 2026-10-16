"""`vdp` command line. Subcommands are added by later tasks/plans."""

from __future__ import annotations

import argparse
import contextlib
import getpass
import sys
from pathlib import Path

from .auth import AuthCorrupt, AuthStore
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


LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def _prompt_new_password(getpass_fn=None, tries: int = 3) -> str | None:
    """Ask for a new password twice; None after `tries` failed attempts (empty, or the two do not match).
    Shared by `vdp set-password` (one try) and `vdp serve`'s menus."""
    getpass_fn = getpass_fn or getpass.getpass
    try:
        for _ in range(tries):
            first = getpass_fn("New password: ")
            if not first:
                print("The password must not be empty.")
                continue
            if getpass_fn("Repeat it: ") != first:
                print("The two passwords do not match.")
                continue
            return first
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled; no password set.")
    return None


def _ask(prompt: str, *, stdin, ask) -> str:
    """A menu answer, lowercased (callers compare it whole); "" when there is no terminal to ask or input ends."""
    if not stdin.isatty():
        return ""
    try:
        return ask(prompt).strip().lower()
    except EOFError:
        return ""


def _set_password(settings: Settings, args, *, stdin=None, getpass_fn=None) -> int:
    store = AuthStore(settings.project.data_root)
    if args.off:
        store.clear()
        print(f"Login is off: removed {store.path}")
        return 0
    if args.stdin:
        password = (stdin or sys.stdin).readline().rstrip("\r\n")
        if not password:
            print("error: no password on stdin")
            return 1
    else:
        password = _prompt_new_password(getpass_fn, tries=1)
        if password is None:
            return 1
    store.set_password(password)
    print(f"Password set in {store.path}. Every existing session has ended.")
    return 0


def _tls_pair(settings: Settings) -> tuple[Path, Path, bool]:
    """(cert, key, supplied_by_user). Raises ValueError when only one of server.cert / server.key is set."""
    from .tls import default_pair

    cert, key = settings.server.cert, settings.server.key
    if (cert is None) != (key is None):
        raise ValueError("set both server.cert and server.key in the config, or neither")
    if cert is not None and key is not None:
        return cert, key, True
    cert, key = default_pair(settings.project.data_root)
    return cert, key, False


def _print_cert(cert: Path) -> None:
    from .tls import cert_info

    info = cert_info(cert)
    print(f"certificate: {info.path}")
    print(f"  names:   {', '.join(info.names)}")
    print(f"  expires: {info.not_after:%Y-%m-%d}")
    print(f"  SHA-256: {info.fingerprint}")


def _cert(settings: Settings, args) -> int:
    from .tls import generate_cert

    try:
        cert, key, supplied = _tls_pair(settings)
    except ValueError as e:
        print(f"error: {e}")
        return 2
    if args.regenerate:
        if supplied:
            print("error: server.cert/server.key are set in the config; replace those files yourself")
            return 2
        generate_cert(cert, key)
        print("Generated a new self-signed certificate.")
    elif not cert.is_file():
        print(f"No certificate at {cert} yet; `vdp serve --https` or `vdp cert --regenerate` creates one.")
        return 1
    _print_cert(cert)
    return 0


def _serve(settings: Settings, host: str | None, port: int | None, https: bool = False, *,
           stdin=None, ask=None, getpass_fn=None, run=None, make_app=None) -> int:
    from .tls import ensure_cert, local_ipv4s

    stdin = stdin or sys.stdin
    ask = ask or input
    host = host or settings.server.host
    port = port or settings.server.port
    network = host not in LOCAL_HOSTS
    store = AuthStore(settings.project.data_root)

    def set_new_password() -> bool:
        password = _prompt_new_password(getpass_fn)
        if password is None:
            print("No password set; quitting.")
            return False
        store.set_password(password)
        return True

    try:
        config = store.load()
    except AuthCorrupt as e:
        answer = _ask(f"auth.json is unreadable ({e}). [s] set a new password, [w] wipe it and start without "
                      "a password, [q] quit (default): ", stdin=stdin, ask=ask)
        if answer == "s":
            if not set_new_password():
                return 1
            config = store.load()
        elif answer == "w":
            store.clear()
            config = None
            print(f"Removed {store.path}.")
        else:
            print(f"error: {e}. Fix it with `vdp set-password` (a new password) or `vdp set-password --off` "
                  "(no login).")
            return 2
    if network and config is None:
        answer = _ask("No password is set, so the server would be open to the network. [s] set one now, "
                      "[q] quit (default): ", stdin=stdin, ask=ask)
        if answer != "s":
            print(f"error: refusing to serve on {host} without a password; run `vdp set-password` first.")
            return 2
        if not set_new_password():
            return 1
        config = store.load()

    ssl: dict = {}
    use_https = https or settings.server.https
    if use_https:
        try:
            cert, key, supplied = _tls_pair(settings)
        except ValueError as e:
            print(f"error: {e}")
            return 2
        if supplied:
            missing = [str(p) for p in (cert, key) if not p.is_file()]
            if missing:
                print(f"error: missing {', '.join(missing)}")
                return 2
        elif ensure_cert(cert, key, [host]):
            print(f"Generated a self-signed certificate: {cert}")
        _print_cert(cert)
        ssl = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    elif network:
        print("WARNING: no HTTPS, so the password and session cookie travel unencrypted; add --https")

    scheme = "https" if use_https else "http"
    shown = (local_ipv4s() or [host]) if host in ("0.0.0.0", "::") else [host]
    for h in shown:
        print(f"open:      {scheme}://{h}:{port}/")
    print(f"login:     {'on' if config is not None else 'off'}")
    print(f"data_root: {settings.project.data_root}")
    print(f"model_dir: {settings.sam3.model_dir}")
    if make_app is None:
        from .api.app import create_app as make_app
    if run is None:
        import uvicorn

        run = uvicorn.run
    import os
    import signal

    # Closing the terminal sends SIGHUP, which uvicorn leaves at its default (die on the spot, skipping the
    # lifespan shutdown that cancels running jobs and lets an augment job remove its half-written export);
    # turn it into SIGTERM, uvicorn's graceful shutdown.
    def _hup_to_term(signum, frame) -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    previous = _install_handlers(_hup_to_term, (signal.SIGHUP,))
    try:
        run(make_app(settings, auth_store=store), host=host, port=port, log_level="info",
            proxy_headers=False, **ssl)  # X-Forwarded-* would let a local client spoof its address and scheme
    finally:
        _restore_handlers(previous)
    return 0


def _install_handlers(handler, sigs) -> dict:
    """Install `handler` for each of `sigs` and return the handlers to put back with `_restore_handlers`.
    Only the main thread may set handlers (elsewhere this does nothing). A SIGHUP that is already ignored is
    left alone: that is `nohup`, asking for the command to outlive its terminal."""
    import signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        return {}
    previous = {}
    for sig in sigs:
        if sig == signal.SIGHUP and signal.getsignal(sig) is signal.SIG_IGN:
            continue
        previous[sig] = signal.signal(sig, handler)
    return previous


def _restore_handlers(previous: dict) -> None:
    import signal

    for sig, handler in previous.items():
        signal.signal(sig, handler)


def _fmt(v: float | None) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def _train(settings: Settings, args, *, gpus=None, worker_cmd=None, poll: float = 2.0) -> int:
    import signal
    import time

    from pydantic import ValidationError

    from .labeling.errors import BadRequest, Conflict
    from .store import NotFound
    from .train.models import DEFAULT_MODELS, TrainParams, find_export, list_exports
    from .train.results import BOX_MAP, MASK_MAP, MASK_MAP50, read_metrics
    from .train.scheduler import (
        WORKER_CMD,
        TrainScheduler,
        default_gpu,
        detect_gpus,
        gpu_max_clock_ghz,
        rank_gpus,
    )

    data_root = settings.project.data_root
    if args.list:
        for e in list_exports(data_root):
            print(f"{e.id}  {e.name or '-'}  {e.project_name}  train {e.train_images}  val {e.val_images}  "
                  f"classes: {', '.join(e.classes)}")
        return 0
    if not args.export:
        print("error: --export is required (see `vdp train --list`)")
        return 2
    if args.from_run is not None:
        if args.model:
            print("error: --model is invalid with --from-run")
            return 2
        if args.mode is None:
            print("error: --mode is required with --from-run (continue or finetune)")
            return 2
    try:
        params = TrainParams(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience,
                             workers=args.workers, seed=args.seed, augment=args.augment,
                             train_images_per_epoch=args.images_per_epoch,
                             val_images_per_epoch=args.val_images_per_epoch)
    except ValidationError as e:
        print(f"error: {e}")
        return 2
    gpus = detect_gpus() if gpus is None else gpus
    if not gpus:
        print("FAIL: no CUDA GPU is available to train on")
        return 1
    try:
        export = find_export(data_root, args.export)
        sched = TrainScheduler(data_root, gpus, worker_cmd=worker_cmd or WORKER_CMD)
    except (NotFound, BadRequest) as e:
        print(f"error: {e}")
        return 2
    if args.gpu is not None:
        gpu = args.gpu
        gpu_desc = f"cuda:{gpu}"
    else:
        ranked = rank_gpus(gpus, gpu_max_clock_ghz(gpus))
        busy = sched.busy_gpus()
        gpu = default_gpu(ranked, busy)
        name = next((g.name for g in gpus if g.index == gpu), "")
        reason = "fastest idle GPU" if gpu not in busy else "fastest GPU, all busy"
        gpu_desc = f"cuda:{gpu} ({name}, {reason})"
    try:
        if args.from_run is not None:
            ids = [sched.submit_from(args.from_run, export, args.mode, gpu,
                                     params if args.mode == "finetune" else None)]
        else:
            ids = sched.submit(export, [(m, gpu) for m in (args.model or DEFAULT_MODELS)], params)
    except (NotFound, BadRequest, Conflict) as e:
        print(f"error: {e}")
        return 2
    print(f"training {len(ids)} model(s) on {gpu_desc} from export {export.id} ({export.project_name}); "
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
    previous_handlers = _install_handlers(_raise_interrupt, (signal.SIGTERM, signal.SIGHUP))

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
        _restore_handlers(previous_handlers)
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


def _backgrounds(settings: Settings, args) -> int:
    from .augment.backgrounds import fetch_pool, pool_dir

    data_root = settings.project.data_root
    last_pct: dict[str, int] = {}

    def progress(phase: str, done: int, total: int) -> None:
        pct = int(done * 100 / total) if total else 100
        if last_pct.get(phase) != pct:
            last_pct[phase] = pct
            print(f"{phase} {pct}%", flush=True)

    try:
        dest = pool_dir(data_root, args.name)
        result = fetch_pool(data_root, args.name, force=args.force, progress=progress)
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}")
        return 1
    if result["skipped"]:
        print(f"skipped: {result['images']} images already in {dest} (use --force)")
    else:
        print(f"OK: {result['images']} images in {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vdp")
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
    p_serve.add_argument("--https", action="store_true", help="serve over HTTPS (self-signed unless configured)")
    p_pw = sub.add_parser("set-password", help="set the team password for `vdp serve` (ends every session)")
    pw_mode = p_pw.add_mutually_exclusive_group()
    pw_mode.add_argument("--off", action="store_true", help="remove the password (login off)")
    pw_mode.add_argument("--stdin", action="store_true", help="read the password from one line of stdin")
    p_cert = sub.add_parser("cert", help="show the HTTPS certificate `vdp serve --https` uses")
    p_cert.add_argument("--regenerate", action="store_true", help="replace the self-signed certificate")
    p_train = sub.add_parser("train", help="train YOLO segmentation (-seg) or detection models on an export")
    p_train.add_argument("--list", action="store_true", help="list the exports that have a yolo_seg dataset")
    p_train.add_argument("--export", default=None, help="export id (a folder name under data/exports/)")
    p_train.add_argument("--model", action="append", default=None,
                         help="model to train, repeatable: a -seg model (masks) or yolo26n/s/m/l/x (boxes; trains on the export's yolo_det when it has one) (default: yolo26s/m-seg and yolo11s/m-seg)")
    p_train.add_argument("--from-run", default=None,
                         help="run id to continue or fine-tune from on --export (needs --mode); "
                              "invalid with --model")
    p_train.add_argument("--mode", choices=["continue", "finetune"], default=None,
                         help="with --from-run: continue its own schedule and params, "
                              "or fine-tune a fresh schedule from its best.pt")
    p_train.add_argument("--gpu", type=int, default=None,
                         help="CUDA index (default: the fastest idle GPU; SAM3's GPU is eligible)")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--imgsz", type=int, default=960)
    p_train.add_argument("--batch", type=int, default=8)
    p_train.add_argument("--patience", type=int, default=30)
    p_train.add_argument("--workers", type=int, default=8)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--augment", choices=["default", "strong"], default="default")
    p_train.add_argument("--images-per-epoch", type=int, default=None,
                         help="train on this many images per epoch, a new slice of the shuffled train set each epoch "
                              "(default: all of them)")
    p_train.add_argument("--val-images-per-epoch", type=int, default=None,
                         help="validate each epoch on a fixed subset of this many val images; the final evaluation "
                              "still uses all of them (default: all of them)")
    p_bg = sub.add_parser("backgrounds", help="manage background image pools for synthetic composites")
    bg_sub = p_bg.add_subparsers(dest="bg_cmd", required=True)
    p_bg_fetch = bg_sub.add_parser("fetch", help="download and extract a background pool")
    p_bg_fetch.add_argument("name", help="pool name (e.g. dtd)")
    p_bg_fetch.add_argument("--force", action="store_true",
                            help="redownload even if images are already present")
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
        return _serve(settings, args.host, args.port, args.https)
    if args.cmd == "set-password":
        return _set_password(settings, args)
    if args.cmd == "cert":
        return _cert(settings, args)
    if args.cmd == "train":
        return _train(settings, args)
    if args.cmd == "backgrounds" and args.bg_cmd == "fetch":
        return _backgrounds(settings, args)
    parser.error(f"unknown command {args.cmd}")
    return 2
