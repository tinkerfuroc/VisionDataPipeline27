from __future__ import annotations

import io
from pathlib import Path

import pytest
import vdp.cli
from vdp.auth import AuthStore, verify_password


def _cfg(tmp_path) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(f"project: {{data_root: {tmp_path / 'data'}}}\n")
    return str(p)


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _serve(tmp_path, *, host="0.0.0.0", answers=(), passwords=(), tty=True, https=False, setup=None):
    """Run `_serve` with scripted menu answers and passwords; uvicorn and the app are stubbed out."""
    settings, _ = vdp.cli._load_settings(_cfg(tmp_path))
    if setup is not None:
        setup(settings)
    answer_it, password_it = iter(answers), iter(passwords)
    runs: list[dict] = []
    stores: list[AuthStore] = []

    def make_app(s, auth_store):
        stores.append(auth_store)
        return "app"

    code = vdp.cli._serve(
        settings, host, 28000, https, stdin=Tty() if tty else io.StringIO(),
        ask=lambda prompt: next(answer_it), getpass_fn=lambda prompt: next(password_it),
        run=lambda app, **kw: runs.append(kw), make_app=make_app,
    )
    if runs:
        assert stores and stores[0].path == settings.project.data_root / "auth.json"
    return code, runs, AuthStore(settings.project.data_root)


def _corrupt(tmp_path) -> Path:
    p = tmp_path / "data" / "auth.json"
    p.parent.mkdir(parents=True)
    p.write_text("{not json")
    return p


def test_set_password_from_stdin_then_off(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("s3cret\n"))
    assert vdp.cli.main(["--config", cfg, "set-password", "--stdin"]) == 0
    store = AuthStore(tmp_path / "data")
    assert verify_password("s3cret", store.load().password_hash)
    assert vdp.cli.main(["--config", cfg, "set-password", "--off"]) == 0
    assert store.load() is None


def test_set_password_prompt_mismatch_exits_1(tmp_path, monkeypatch):
    answers = iter(["one", "two"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    assert vdp.cli.main(["--config", _cfg(tmp_path), "set-password"]) == 1
    assert AuthStore(tmp_path / "data").load() is None


def test_cert_command_reports_and_regenerates(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    assert vdp.cli.main(["--config", cfg, "cert"]) == 1  # nothing generated yet
    assert vdp.cli.main(["--config", cfg, "cert", "--regenerate"]) == 0
    out = capsys.readouterr().out
    assert "SHA-256:" in out and str(tmp_path / "data" / "tls" / "cert.pem") in out


def test_serve_locally_without_a_password_starts_without_asking(tmp_path):
    code, runs, _ = _serve(tmp_path, host="127.0.0.1")  # no scripted answers: asking would raise StopIteration
    assert code == 0 and len(runs) == 1


def test_serve_never_trusts_proxy_headers(tmp_path):
    _, runs, _ = _serve(tmp_path, host="127.0.0.1")
    assert runs[0]["proxy_headers"] is False


def test_serve_on_the_network_without_a_password_offers_to_set_one(tmp_path):
    code, runs, store = _serve(tmp_path, answers=["s"], passwords=["pw", "pw"])
    assert code == 0 and len(runs) == 1 and verify_password("pw", store.load().password_hash)


@pytest.mark.parametrize("answer", ["q", "", "x", "stop"])
def test_serve_on_the_network_without_a_password_otherwise_exits_2(tmp_path, answer):
    code, runs, store = _serve(tmp_path, answers=[answer])
    assert code == 2 and runs == [] and store.load() is None


def test_serve_on_the_network_without_a_terminal_exits_2(tmp_path):
    code, runs, _ = _serve(tmp_path, tty=False)
    assert code == 2 and runs == []


def test_serve_with_a_password_but_no_https_warns(tmp_path, capsys):
    AuthStore(tmp_path / "data").set_password("pw")
    code, runs, _ = _serve(tmp_path)
    assert code == 0 and runs and "WARNING: no HTTPS" in capsys.readouterr().out


def test_serve_https_generates_and_uses_a_certificate(tmp_path, capsys):
    code, runs, _ = _serve(tmp_path, host="127.0.0.1", https=True)
    assert code == 0
    assert Path(runs[0]["ssl_certfile"]) == tmp_path / "data" / "tls" / "cert.pem"
    assert Path(runs[0]["ssl_keyfile"]).is_file()
    out = capsys.readouterr().out
    assert "SHA-256:" in out and "https://127.0.0.1:28000/" in out


def test_serve_https_with_a_missing_configured_cert_exits_2(tmp_path, capsys):
    def own_pair(s):
        s.server.cert, s.server.key = tmp_path / "mine.pem", tmp_path / "mine.key"

    code, runs, _ = _serve(tmp_path, host="127.0.0.1", https=True, setup=own_pair)
    assert code == 2 and runs == [] and "mine.pem" in capsys.readouterr().out


def test_serve_https_with_only_one_of_cert_and_key_exits_2(tmp_path):
    def half(s):
        s.server.cert = tmp_path / "mine.pem"

    code, runs, _ = _serve(tmp_path, host="127.0.0.1", https=True, setup=half)
    assert code == 2 and runs == []


def test_corrupt_auth_set_a_new_password_starts_even_on_the_network(tmp_path):
    _corrupt(tmp_path)
    code, runs, store = _serve(tmp_path, answers=["s"], passwords=["pw", "pw"])
    assert code == 0 and runs and verify_password("pw", store.load().password_hash)


def test_corrupt_auth_set_password_gives_up_after_three_failed_tries(tmp_path):
    p = _corrupt(tmp_path)
    code, runs, _ = _serve(tmp_path, answers=["s"], passwords=["a", "b", "", "c", "d"])
    assert code == 1 and runs == [] and p.read_text() == "{not json"


def test_corrupt_auth_wipe_starts_locally(tmp_path):
    p = _corrupt(tmp_path)
    code, runs, _ = _serve(tmp_path, host="127.0.0.1", answers=["w"])
    assert code == 0 and runs and not p.exists()


def test_corrupt_auth_wipe_on_the_network_then_offers_a_password(tmp_path):
    p = _corrupt(tmp_path)
    code, runs, _ = _serve(tmp_path, answers=["w", "q"])
    assert code == 2 and runs == [] and not p.exists()


@pytest.mark.parametrize("answers, tty", [(["q"], True), ([""], True), (["z"], True), (["what?"], True), (["stop"], True), ([], False)])
def test_corrupt_auth_otherwise_exits_2_and_leaves_the_file(tmp_path, answers, tty):
    p = _corrupt(tmp_path)
    code, runs, _ = _serve(tmp_path, host="127.0.0.1", answers=answers, tty=tty)
    assert code == 2 and runs == [] and p.read_text() == "{not json"


@pytest.mark.parametrize("exc", [KeyboardInterrupt, EOFError])
def test_set_password_cancelled_at_the_prompt_exits_1_and_writes_nothing(tmp_path, monkeypatch, capsys, exc):
    def cancel(prompt=""):
        raise exc

    monkeypatch.setattr("getpass.getpass", cancel)
    assert vdp.cli.main(["--config", _cfg(tmp_path), "set-password"]) == 1
    assert "Cancelled; no password set." in capsys.readouterr().out
    assert AuthStore(tmp_path / "data").load() is None


def test_serve_turns_sighup_into_sigterm_so_uvicorn_shuts_down_cleanly(tmp_path):
    # closing the terminal sends SIGHUP, which uvicorn ignores (the process just dies, skipping the lifespan
    # shutdown that cancels jobs); SIGTERM is uvicorn's graceful shutdown
    import os
    import signal
    import time

    settings, _ = vdp.cli._load_settings(_cfg(tmp_path))
    before = signal.getsignal(signal.SIGHUP)
    got: list[int] = []

    def run(app, **kw):
        previous = signal.signal(signal.SIGTERM, lambda signum, frame: got.append(signum))
        try:
            os.kill(os.getpid(), signal.SIGHUP)
            deadline = time.monotonic() + 10
            while not got and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            signal.signal(signal.SIGTERM, previous)

    code = vdp.cli._serve(settings, "127.0.0.1", 28000, False, stdin=Tty(), ask=lambda p: "q",
                          getpass_fn=lambda p: "", run=run, make_app=lambda s, auth_store: "app")
    assert code == 0 and got == [signal.SIGTERM]
    assert signal.getsignal(signal.SIGHUP) == before
