"""Team password for `vdp serve`: password hashing, the auth file, session tokens, login rate limiting.

Standard library only. `<data_root>/auth.json` holds a salted scrypt hash of the one shared password and a random
secret that signs session tokens. Every `set_password` writes a new secret, which ends every existing session.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import os
import secrets
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

AUTH_FILE = "auth.json"
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1
SALT_BYTES, KEY_BYTES, SECRET_BYTES = 16, 32, 32
SESSION_MAX_AGE = 30 * 24 * 3600
FUTURE_TOLERANCE = 60  # seconds a token may claim to be issued in the future (clock skew)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(SALT_BYTES)
    key = hashlib.scrypt(
        password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=KEY_BYTES
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(key)}"


def verify_password(password: str, stored: str) -> bool:
    """False for a wrong password and for anything that is not a well-formed scrypt hash."""
    try:
        scheme, n, r, p, salt, key = stored.split("$")
        if scheme != "scrypt":
            return False
        want = base64.b64decode(key, validate=True)
        got = hashlib.scrypt(
            password.encode(),
            salt=base64.b64decode(salt, validate=True),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(want),
        )
    except ValueError:  # bad split, int(), base64 (binascii.Error), or scrypt parameters
        return False
    return hmac.compare_digest(got, want)


def _sign(secret: bytes, payload: str) -> str:
    return hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()


def make_token(secret: bytes, now: float | None = None) -> str:
    payload = f"v1.{int(time.time() if now is None else now)}"
    return f"{payload}.{_sign(secret, payload)}"


def check_token(secret: bytes, token: str, now: float | None = None) -> bool:
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != "v1" or not parts[1].isdigit():
        return False
    if not hmac.compare_digest(_sign(secret, f"v1.{parts[1]}"), parts[2]):
        return False
    age = (time.time() if now is None else now) - int(parts[1])
    return -FUTURE_TOLERANCE <= age <= SESSION_MAX_AGE


class AuthCorrupt(Exception):
    """`auth.json` exists but cannot be used; the message says why."""


@dataclass(frozen=True)
class AuthConfig:
    password_hash: str
    secret: bytes


_UNSET = object()


class AuthStore:
    def __init__(self, data_root: Path):
        self.path = Path(data_root) / AUTH_FILE
        self._seen: object = _UNSET
        self._cached: AuthConfig | AuthCorrupt | None = None

    def load(self) -> AuthConfig | None:
        """The current config, or None when no password is set. Raises `AuthCorrupt`."""
        try:
            raw = self.path.read_text()
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as e:
            raise AuthCorrupt(f"cannot read {self.path}: {e}") from e
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise AuthCorrupt(f"{self.path} is not valid JSON ({e.msg})") from e
        if not isinstance(data, dict) or data.get("version") != 1:
            raise AuthCorrupt(f"{self.path} has an unknown format")
        password_hash = data.get("password_hash")
        if (
            not isinstance(password_hash, str)
            or not password_hash.startswith("scrypt$")
            or password_hash.count("$") != 5
        ):
            raise AuthCorrupt(f"{self.path} has no valid password hash")
        try:
            secret = bytes.fromhex(data["secret"]) if isinstance(data.get("secret"), str) else b""
        except ValueError:
            secret = b""
        if len(secret) != SECRET_BYTES:
            raise AuthCorrupt(f"{self.path} has no valid secret")
        return AuthConfig(password_hash, secret)

    def current(self) -> AuthConfig | AuthCorrupt | None:
        """`load()` for the request path: re-reads only when the file's identity changed, and returns (rather
        than raises) `AuthCorrupt`, which the gate treats as "login on, no session valid" (fail closed)."""
        try:
            st = self.path.stat()
            seen: object = (st.st_ino, st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            seen = None
        except OSError as e:
            seen = ("unreadable", str(e))
        if seen != self._seen:
            try:
                self._cached = self.load()
            except AuthCorrupt as e:
                log.warning("login: %s; refusing every session until `vdp set-password` rewrites it", e)
                self._cached = e
            self._seen = seen
        return self._cached

    def set_password(self, password: str) -> None:
        if not password:
            raise ValueError("the password must not be empty")
        data = {
            "version": 1,
            "password_hash": hash_password(password),
            "secret": secrets.token_hex(SECRET_BYTES),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{AUTH_FILE}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.chmod(tmp, 0o600)  # O_CREAT's mode is masked by the umask and ignored for an existing file
        os.replace(tmp, self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


class LoginLimiter:
    """Wrong passwords per client address in a sliding window. In memory, so a restart forgets it."""

    def __init__(
        self, max_failures: int = 5, window_s: float = 300.0, clock: Callable[[], float] = time.monotonic
    ):
        self.max_failures = max_failures
        self.window_s = window_s
        self.clock = clock
        self._failures: dict[str, deque[float]] = {}

    def _prune(self, key: str) -> deque[float] | None:
        q = self._failures.get(key)
        if q is None:
            return None
        cutoff = self.clock() - self.window_s
        while q and q[0] <= cutoff:
            q.popleft()
        if not q:
            del self._failures[key]
            return None
        return q

    def retry_after(self, key: str) -> int:
        """Seconds until `key` may try again; 0 = allowed now."""
        q = self._prune(key)
        if q is None or len(q) < self.max_failures:
            return 0
        return max(1, math.ceil(q[0] + self.window_s - self.clock()))

    def failed(self, key: str) -> None:
        self._prune(key)
        self._failures.setdefault(key, deque()).append(self.clock())

    def succeeded(self, key: str) -> None:
        self._failures.pop(key, None)
