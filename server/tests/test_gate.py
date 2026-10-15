from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from vdp.api.app import create_app
from vdp.api.gate import COOKIE, safe_next
from vdp.auth import AuthStore
from vdp.config import Settings
from vdp.engine.fake import FakeEngine

PASSWORD = "team pass"


def _client(tmp_path, password=PASSWORD):
    s = Settings.load(None)
    s.project.data_root = tmp_path / "data"
    store = AuthStore(s.project.data_root)
    if password is not None:
        store.set_password(password)
    return TestClient(create_app(s, engine=FakeEngine(), load_engine=False)), store


@pytest.fixture
def gated(tmp_path):
    client, store = _client(tmp_path)
    with client:
        yield client, store


def _login(client, password=PASSWORD, next_="/"):
    return client.post("/api/auth/login", data={"password": password, "next": next_}, follow_redirects=False)


def _ws_close_code(client, path="/ws/jobs/nope"):
    with client.websocket_connect(path) as ws, pytest.raises(WebSocketDisconnect) as closed:
        ws.receive_text()
    return closed.value.code


def test_login_off_passes_everything(tmp_path):
    client, _ = _client(tmp_path, password=None)
    with client:
        assert client.get("/api/projects").status_code == 200
        assert client.get("/api/health").json()["auth"] == {"enabled": False}
        assert _ws_close_code(client) == 4404  # the jobs route's own answer: unknown job
        r = client.get("/login?next=/train", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/train"
        assert _login(client, password="anything", next_="/train").headers["location"] == "/train"


def test_without_a_session_api_is_401_pages_redirect_and_sockets_close(gated):
    client, _ = gated
    r = client.get("/api/projects")
    assert r.status_code == 401 and r.json() == {"detail": "login required"}
    assert client.get("/api/health").status_code == 401
    assert client.get("/openapi.json").status_code == 401
    assert client.get("/docs").status_code == 401
    r = client.get("/label/c1?f=3", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login?next=%2Flabel%2Fc1%3Ff%3D3"
    assert _ws_close_code(client) == 4401
    page = client.get("/login")
    assert page.status_code == 200 and 'name="password"' in page.text and "Wrong password" not in page.text


def test_login_sets_a_cookie_that_opens_the_api_and_sockets(gated):
    client, _ = gated
    r = _login(client, next_="/train")
    assert r.status_code == 303 and r.headers["location"] == "/train"
    cookie = r.headers["set-cookie"].lower()
    assert f"{COOKIE}=" in cookie and "httponly" in cookie and "samesite=strict" in cookie
    assert "max-age=2592000" in cookie and "secure" not in cookie  # plain-HTTP test client
    assert client.get("/api/projects").status_code == 200
    assert client.get("/api/health").json()["auth"] == {"enabled": True}
    assert _ws_close_code(client) == 4404
    assert client.get("/login?next=/train", follow_redirects=False).headers["location"] == "/train"


def test_wrong_password_redirects_with_an_error_and_sets_no_cookie(gated):
    client, _ = gated
    r = _login(client, password="nope", next_="/train")
    assert r.status_code == 303 and r.headers["location"] == "/login?error=wrong&next=%2Ftrain"
    assert COOKIE not in client.cookies
    assert "Wrong password." in client.get("/login?error=wrong").text


def test_the_sixth_attempt_within_five_minutes_is_429_even_with_the_right_password(gated):
    client, _ = gated
    for _ in range(5):
        assert _login(client, password="nope").status_code == 303
    r = _login(client)
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    assert "Too many attempts" in r.text
    assert COOKIE not in client.cookies


def test_logout_clears_the_session(gated):
    client, _ = gated
    _login(client)
    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/projects").status_code == 401


def test_a_new_password_ends_existing_sessions(gated):
    client, store = gated
    _login(client)
    store.set_password("another")
    assert client.get("/api/projects").status_code == 401
    assert _login(client, password="another").status_code == 303
    assert client.get("/api/projects").status_code == 200


def test_auth_json_corrupted_while_running_fails_closed(gated):
    client, store = gated
    _login(client)
    store.path.write_text("{not json")
    assert client.get("/api/projects").status_code == 401
    assert _login(client).headers["location"].startswith("/login?error=wrong")


def test_a_malformed_cookie_header_counts_as_logged_out(gated):
    client, _ = gated
    r = client.get("/api/projects", headers={"cookie": 'vdp_session="unterminated; =;;'})
    assert r.status_code == 401


@pytest.mark.parametrize("value, expected", [
    ("/train?x=1", "/train?x=1"), ("", "/"), (None, "/"), ("//evil.example", "/"), ("/\\evil.example", "/"),
    ("https://evil.example", "/"), ("/\t/evil.example", "/"), ("/a\nb", "/"), ("/login?next=/x", "/"),
    ("/#/upload?project=x", "/#/upload?project=x"),
])
def test_safe_next(value, expected):
    assert safe_next(value) == expected


def test_the_login_redirect_target_is_sanitised(gated):
    client, _ = gated
    assert _login(client, next_="//evil.example").headers["location"] == "/"
