"""The login gate in front of `vdp serve` (spec: LAN access with a team password and HTTPS).

`AuthGate` is a pure ASGI middleware, so it sees WebSocket handshakes as well as HTTP requests. With no password set
it passes everything through. With one, a request needs a valid `vdp_session` cookie, except `OPEN_PATHS`: API and
docs requests get 401, page loads a redirect to `/login`, and a WebSocket is accepted and closed with 4401 (a socket
closed before accept reaches the browser as a failed handshake, code 1006, indistinguishable from a network error).
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from http.cookies import CookieError, SimpleCookie
from urllib.parse import quote

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from ..auth import (
    SESSION_MAX_AGE,
    AuthConfig,
    AuthStore,
    LoginLimiter,
    check_token,
    make_token,
    verify_password,
)

COOKIE = "vdp_session"
LOGIN_REQUIRED = "login required"
WS_LOGIN_REQUIRED = 4401
OPEN_PATHS = frozenset({"/login", "/api/auth/login", "/api/auth/logout", "/favicon.ico"})


@dataclass
class GateState:
    store: AuthStore
    limiter: LoginLimiter = field(default_factory=LoginLimiter)

    def enabled(self) -> bool:
        return self.store.current() is not None  # a corrupt file counts as on (fail closed)

    def valid(self, token: str | None) -> bool:
        cfg = self.store.current()
        return isinstance(cfg, AuthConfig) and token is not None and check_token(cfg.secret, token)


def safe_next(value: str | None) -> str:
    """Where to go after logging in: a path on this server only. Rejects `//host`, backslashes and control
    characters (browsers strip tabs/newlines and read `\\` as `/`, turning `/\\t/evil` into `//evil`), and the
    login page itself."""
    if not value or not value.startswith("/") or value.startswith(("//", "/login")):
        return "/"
    if "\\" in value or any(ord(c) < 0x20 or ord(c) == 0x7F for c in value):
        return "/"
    return value


def _session_cookie(scope) -> str | None:
    for name, value in scope.get("headers", []):
        if name != b"cookie":
            continue
        jar = SimpleCookie()
        try:
            jar.load(value.decode("latin-1"))
        except CookieError:
            continue
        if COOKIE in jar:
            return jar[COOKIE].value
    return None


def _wants_json(path: str) -> bool:
    return path.startswith(("/api/", "/docs")) or path in ("/redoc", "/openapi.json")


class AuthGate:
    def __init__(self, app, state: GateState):
        self.app = app
        self.state = state

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or not self.state.enabled() \
                or self.state.valid(_session_cookie(scope)):
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        if scope["type"] == "websocket":
            await receive()  # websocket.connect
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.close", "code": WS_LOGIN_REQUIRED})
            return
        if path in OPEN_PATHS:
            await self.app(scope, receive, send)
            return
        if _wants_json(path):
            response: Response = JSONResponse({"detail": LOGIN_REQUIRED}, status_code=401)
        else:
            query = scope.get("query_string", b"").decode("latin-1")
            target = path + (f"?{query}" if query else "")
            response = RedirectResponse(f"/login?next={quote(target, safe='')}", status_code=303)
        await response(scope, receive, send)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Log in · VisionDataPipeline27</title>
<style>
body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#2B2D30;color:#E9E7E2;
font:15px/1.4 Barlow,system-ui,sans-serif}}
.box{{width:min(320px,calc(100vw - 32px));box-sizing:border-box;background:#36383C;border:1px solid #4A4D52;
border-radius:6px;padding:20px;display:flex;flex-direction:column;gap:12px}}
h1{{margin:0;font-size:17px;font-weight:600}}
label{{display:flex;flex-direction:column;gap:4px;color:#A3A5AA;font-size:13px}}
input{{background:#1E1F21;color:#E9E7E2;border:1px solid #4A4D52;border-radius:4px;padding:7px 8px;font:inherit}}
input:focus,button:focus{{outline:2px solid #E9E7E2;outline-offset:1px}}
button{{background:#E9E7E2;color:#1E1F21;border:0;border-radius:4px;padding:8px;font:inherit;font-weight:600;
cursor:pointer}}
.error{{color:#FF8A80;margin:0;font-size:13px}}
a{{color:#E9E7E2}}
</style></head><body>{body}</body></html>"""


def render_login(next_path: str, wrong: bool) -> str:
    error = '<p class="error" role="alert">Wrong password.</p>' if wrong else ""
    return _PAGE.format(body=(
        f'<form class="box" method="post" action="/api/auth/login"><h1>VisionDataPipeline27</h1>{error}'
        '<label>Team password<input type="password" name="password" autocomplete="current-password" '
        'autofocus required></label>'
        f'<input type="hidden" name="next" value="{html.escape(next_path)}">'
        '<button type="submit">Log in</button></form>'
    ))


def render_message(text: str) -> str:
    return _PAGE.format(body=(
        f'<div class="box"><h1>VisionDataPipeline27</h1><p class="error" role="alert">{html.escape(text)}</p>'
        '<a href="/login">Back to log in</a></div>'
    ))


router = APIRouter()


def _gate(request: Request) -> GateState:
    return request.app.state.gate


@router.get("/login", include_in_schema=False)
def login_page(request: Request, next_: str = Query("/", alias="next"), error: str | None = None) -> Response:
    gate = _gate(request)
    target = safe_next(next_)
    if not gate.enabled() or gate.valid(request.cookies.get(COOKIE)):
        return RedirectResponse(target, status_code=303)
    return HTMLResponse(render_login(target, error == "wrong"))


@router.post("/api/auth/login", include_in_schema=False)
def login(request: Request, password: str = Form(""), next_: str = Form("/", alias="next")) -> Response:
    gate = _gate(request)
    target = safe_next(next_)
    cfg = gate.store.current()
    if cfg is None:
        return RedirectResponse(target, status_code=303)  # login is off: nothing to log into
    client = request.client.host if request.client else "unknown"
    wait = gate.limiter.retry_after(client)
    if wait:
        return HTMLResponse(render_message(f"Too many attempts. Try again in {wait} s."), status_code=429,
                            headers={"Retry-After": str(wait)})
    if not isinstance(cfg, AuthConfig) or not verify_password(password, cfg.password_hash):
        gate.limiter.failed(client)
        return RedirectResponse(f"/login?error=wrong&next={quote(target, safe='')}", status_code=303)
    gate.limiter.succeeded(client)
    response = RedirectResponse(target, status_code=303)
    response.set_cookie(COOKIE, make_token(cfg.secret), max_age=SESSION_MAX_AGE, path="/", httponly=True,
                        samesite="strict", secure=request.url.scheme == "https")
    return response


@router.post("/api/auth/logout", status_code=204, include_in_schema=False)
def logout(request: Request) -> Response:
    response = Response(status_code=204)
    response.delete_cookie(COOKIE, path="/", httponly=True, samesite="strict",
                           secure=request.url.scheme == "https")
    return response
