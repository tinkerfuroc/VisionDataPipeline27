from __future__ import annotations

import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from vdp.auth import (
    SESSION_MAX_AGE,
    AuthCorrupt,
    AuthStore,
    LoginLimiter,
    check_token,
    hash_password,
    make_token,
    verify_password,
)

SECRET = bytes(range(32))


def test_hash_round_trip_and_wrong_password():
    h = hash_password("correct horse")
    assert h.startswith("scrypt$16384$8$1$")
    assert verify_password("correct horse", h)
    assert not verify_password("correct horsE", h)


def test_hashes_are_salted():
    assert hash_password("x") != hash_password("x")


@pytest.mark.parametrize("password", ["pässwörd 密码 🔒", "x" * 1000])
def test_unicode_and_long_passwords_round_trip(password):
    assert verify_password(password, hash_password(password))


@pytest.mark.parametrize("stored", [
    "", "scrypt$1$2", "bcrypt$16384$8$1$AAAA$AAAA", "scrypt$x$8$1$AAAA$AAAA", "scrypt$16384$8$1$!!$AAAA",
    "scrypt$3$8$1$AAAA$AAAA",
])
def test_malformed_hashes_never_verify(stored):
    assert not verify_password("x", stored)


def test_token_is_valid_for_30_days():
    t = make_token(SECRET, now=1_000_000)
    assert check_token(SECRET, t, now=1_000_000)
    assert check_token(SECRET, t, now=1_000_000 + SESSION_MAX_AGE)
    assert not check_token(SECRET, t, now=1_000_000 + SESSION_MAX_AGE + 1)


def test_token_from_more_than_60_s_in_the_future_is_rejected():
    t = make_token(SECRET, now=1_000_100)
    assert check_token(SECRET, t, now=1_000_100 - 60)
    assert not check_token(SECRET, t, now=1_000_100 - 61)


def test_tampered_foreign_and_junk_tokens_are_rejected():
    t = make_token(SECRET, now=1_000_000)
    version, issued, mac = t.split(".")
    assert not check_token(SECRET, f"{version}.{int(issued) + 1}.{mac}", now=1_000_000)
    assert not check_token(bytes(32), t, now=1_000_000)
    for junk in ["", "v1", "v1.abc.def", f"v2.{issued}.{mac}", t + "x", t + ".x"]:
        assert not check_token(SECRET, junk, now=1_000_000)


@pytest.mark.parametrize("bad_token", [
    "v1.5.é",
    "v1.5.\udc80",
    "v1.².0000000000000000000000000000000000000000000000000000000000000000",
])
def test_non_ascii_tokens_return_false(bad_token):
    """Non-ASCII tokens must return False, never raise TypeError."""
    assert not check_token(SECRET, bad_token, now=5)


def test_token_with_non_ascii_mac_char_returns_false():
    """Valid token format but with one MAC character replaced by non-ASCII."""
    t = make_token(SECRET, now=1_000_000)
    version, issued, mac = t.split(".")
    # Replace last character of MAC with non-ASCII
    bad_token = f"{version}.{issued}.{mac[:-1]}é"
    assert not check_token(SECRET, bad_token, now=1_000_000)


def test_store_without_a_file_has_no_password(tmp_path):
    store = AuthStore(tmp_path / "data")
    assert store.load() is None
    assert store.current() is None


def test_set_password_creates_missing_dirs_and_a_private_file(tmp_path):
    store = AuthStore(tmp_path / "data" / "nested")
    store.set_password("pw")
    assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600
    cfg = store.load()
    assert verify_password("pw", cfg.password_hash) and len(cfg.secret) == 32
    text = store.path.read_text()
    assert json.loads(text)["version"] == 1 and '"pw"' not in text


def test_a_new_password_rotates_the_secret(tmp_path):
    store = AuthStore(tmp_path)
    store.set_password("a")
    first = store.load().secret
    store.set_password("a")
    assert store.load().secret != first


def test_set_password_rejects_an_empty_password(tmp_path):
    with pytest.raises(ValueError):
        AuthStore(tmp_path).set_password("")


def test_clear_removes_the_file_and_is_idempotent(tmp_path):
    store = AuthStore(tmp_path)
    store.set_password("a")
    store.clear()
    assert store.load() is None
    store.clear()


def test_current_follows_immediate_rewrites(tmp_path):
    store = AuthStore(tmp_path)
    assert store.current() is None
    store.set_password("a")
    assert verify_password("a", store.current().password_hash)
    store.set_password("b")  # within the same second: identity is inode + mtime_ns + size
    assert verify_password("b", store.current().password_hash)
    store.clear()
    assert store.current() is None


@pytest.mark.parametrize("content, reason", [
    ("{not json", "not valid JSON"),
    ("[]", "unknown format"),
    ('{"version": 2}', "unknown format"),
    ('{"version": 1, "password_hash": "x", "secret": "00"}', "password hash"),
    (json.dumps({"version": 1, "password_hash": "scrypt$16384$8$1$AAAA$AAAA", "secret": "zz"}), "secret"),
])
def test_corrupt_files_raise_from_load_and_are_returned_by_current(tmp_path, content, reason):
    store = AuthStore(tmp_path)
    store.path.write_text(content)
    with pytest.raises(AuthCorrupt, match=reason):
        store.load()
    assert isinstance(store.current(), AuthCorrupt)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_limiter_blocks_after_five_attempts_until_the_window_passes():
    clock = Clock()
    limiter = LoginLimiter(clock=clock)
    for _ in range(5):
        assert limiter.attempt("a") == 0
        clock.t += 1
    assert limiter.attempt("a") == 295  # first attempt at t=0, window 300 s, now t=5
    assert limiter.retry_after("a") == 295
    assert limiter.attempt("b") == 0  # per address
    clock.t = 300.0
    assert limiter.attempt("a") == 0  # the t=0 attempt has left the window


def test_a_success_clears_the_count():
    limiter = LoginLimiter(clock=Clock())
    for _ in range(4):
        assert limiter.attempt("a") == 0
    limiter.succeeded("a")
    for _ in range(5):
        assert limiter.attempt("a") == 0
    assert limiter.attempt("a") > 0


def test_concurrent_attempts_are_counted_exactly():
    limiter = LoginLimiter()
    barrier = threading.Barrier(20)

    def go() -> int:
        barrier.wait()
        return limiter.attempt("a")

    with ThreadPoolExecutor(20) as pool:
        results = list(pool.map(lambda _: go(), range(20)))
    assert results.count(0) == 5
