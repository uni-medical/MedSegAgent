"""Exercise browser auth through the real HTTP adapter and OAuth exchange boundary."""

import hashlib
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from starlette.testclient import TestClient

from medsegagent.auth import OAUTH_COOKIE, SESSION_COOKIE
from medsegagent.web import create_app


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("MEDSEGAGENT_GITHUB_CLIENT_ID", raising=False)
    monkeypatch.delenv("MEDSEGAGENT_GITHUB_CLIENT_SECRET", raising=False)
    return create_app(tmp_path, "https://testserver")


def test_guest_cookie_restart_logout_and_legacy_credentials(app):
    db = app.state.service.db
    legacy = "old-cookie"
    with db:
        db.execute(
            "INSERT INTO sessions VALUES(?,?,?)",
            (hashlib.sha256(legacy.encode()).hexdigest(), "old", time.time() + 3600),
        )
    with TestClient(app, base_url="https://testserver") as client:
        assert client.get("/api/session").json() == {
            "authenticated": False,
            "github_enabled": False,
            "identity": None,
        }
        for headers in (
            {"Authorization": "Bearer " + "a" * 40},
            {"Cookie": "medseg_session=" + legacy},
        ):
            assert client.get("/api/tasks", headers=headers).status_code == 401
            assert client.get("/api/session", headers=headers).json()["authenticated"] is False
        assert client.post("/api/session", json={"token": "a" * 40}).status_code == 405
        guest = client.post("/api/auth/guest")
        assert guest.json()["identity"] == {"kind": "guest", "display_name": "游客"}
        cookie = client.cookies.get(SESSION_COOKIE)
        assert all(
            x in guest.headers["set-cookie"].lower() for x in ("httponly", "secure", "samesite=lax")
        )
        assert client.post("/api/auth/guest").json() == guest.json()
        assert client.cookies.get(SESSION_COOKIE) == cookie
        assert client.get("/api/tasks").json() == []
        assert client.delete("/api/session").json()["authenticated"] is False
        assert (
            client.get("/api/tasks", headers={"Cookie": SESSION_COOKIE + "=" + cookie}).status_code
            == 401
        )
        assert db.execute("SELECT principal FROM sessions").fetchone()[0] == "old"


@pytest.mark.parametrize(
    "path,method",
    [("/api/auth/guest", "post"), ("/api/session", "delete"), ("/api/auth/github/start", "get")],
)
def test_auth_cross_origin_rejected(app, path, method):
    with TestClient(app, base_url="https://testserver") as client:
        for headers in ({"Origin": "https://evil.invalid"}, {"Sec-Fetch-Site": "cross-site"}):
            assert getattr(client, method)(path, headers=headers).status_code == 403


def oauth_app(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDSEGAGENT_GITHUB_CLIENT_ID", "test-client")
    monkeypatch.setenv("MEDSEGAGENT_GITHUB_CLIENT_SECRET", "test-secret")
    app = create_app(tmp_path, "https://testserver")
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path == "/login/oauth/access_token":
            fields = parse_qs(request.content.decode())
            assert fields["redirect_uri"] == ["https://testserver/api/auth/github/callback"]
            assert len(fields["code_verifier"][0]) >= 43
            return httpx.Response(
                200, json={"access_token": "provider-only", "token_type": "bearer"}
            )
        assert request.url == "https://api.github.com/user"
        return httpx.Response(200, json={"id": 123, "login": "test-user", "name": "Test User"})

    app.state.auth._transport = httpx.MockTransport(handler)
    return app, calls


def start(client):
    response = client.get("/api/auth/github/start", follow_redirects=False)
    assert response.status_code == 303
    query = parse_qs(urlsplit(response.headers["location"]).query, keep_blank_values=True)
    assert query["scope"] == [""]
    assert query["code_challenge_method"] == ["S256"]
    assert "code_verifier" not in query
    assert client.cookies.get(OAUTH_COOKIE)
    return query["state"][0]


def test_github_login_replay_and_stable_account(tmp_path, monkeypatch):
    app, calls = oauth_app(tmp_path, monkeypatch)
    with TestClient(app, base_url="https://testserver") as client:
        client.post("/api/auth/guest")
        guest_cookie = client.cookies.get(SESSION_COOKIE)
        state = start(client)
        callback = "/api/auth/github/callback?state=" + state + "&code=code-one"
        success = client.get(callback, follow_redirects=False)
        assert success.headers["location"] == "/"
        assert not client.cookies.get(OAUTH_COOKIE)
        assert app.state.auth.authenticate(guest_cookie) is None
        session = client.get("/api/session").json()
        assert session["identity"] == {"kind": "github", "display_name": "Test User"}
        first = app.state.auth.authenticate(client.cookies.get(SESSION_COOKIE)).id
        assert first == "github:123"
        assert client.get(callback, follow_redirects=False).headers["location"] == "/?error=oauth"
        assert len(calls) == 2
        client.delete("/api/session")
        state = start(client)
        assert (
            client.get(
                "/api/auth/github/callback",
                params={"state": state, "code": "code-two"},
                follow_redirects=False,
            ).headers["location"]
            == "/"
        )
        assert app.state.auth.authenticate(client.cookies.get(SESSION_COOKIE)).id == first
        rows = app.state.service.db.execute("SELECT * FROM auth_sessions").fetchall()
        assert all("provider-only" not in str(tuple(row)) for row in rows)


@pytest.mark.parametrize("bad", ["denied", "duplicate-state", "duplicate-code", "other-browser"])
def test_oauth_invalid_callback_never_exchanges(tmp_path, monkeypatch, bad):
    app, calls = oauth_app(tmp_path, monkeypatch)
    with TestClient(app, base_url="https://testserver") as client:
        state = start(client)
        params = [("state", state), ("code", "one")]
        if bad == "denied":
            params.append(("error", "access_denied"))
        if bad == "duplicate-state":
            params.append(("state", state))
        if bad == "duplicate-code":
            params.append(("code", "two"))
        if bad == "other-browser":
            client.cookies.clear()
        response = client.get("/api/auth/github/callback", params=params, follow_redirects=False)
        assert response.headers["location"] == "/?error=oauth"
        assert not calls
        assert client.get("/api/session").json()["authenticated"] is False


def test_guest_creation_burst_limit_does_not_limit_existing_sessions(app):
    with TestClient(app, base_url="https://testserver") as client:
        first = client.post("/api/auth/guest")
        cookie = client.cookies.get(SESSION_COOKIE)
        assert first.status_code == 200
        for _ in range(119):
            client.cookies.clear()
            assert client.post("/api/auth/guest").status_code == 200
        client.cookies.clear()
        assert client.post("/api/auth/guest").status_code == 429
        assert (
            client.post(
                "/api/auth/guest", headers={"Cookie": SESSION_COOKIE + "=" + cookie}
            ).status_code
            == 200
        )
        assert (
            client.get("/api/tasks", headers={"Cookie": SESSION_COOKIE + "=" + cookie}).status_code
            == 200
        )


def test_head_never_submits_or_consumes_oauth(tmp_path, monkeypatch):
    app, calls = oauth_app(tmp_path, monkeypatch)
    with TestClient(app, base_url="https://testserver") as client:
        client.post("/api/auth/guest")

        def forbidden(*args, **kwargs):
            raise AssertionError("HEAD must not submit tasks")

        monkeypatch.setattr(app.state.service, "submit", forbidden)
        response = client.request(
            "HEAD",
            "/api/tasks",
            json={"text": "liver"},
            headers={"Origin": "https://evil.invalid", "Sec-Fetch-Site": "cross-site"},
        )
        assert response.status_code == 200
        assert response.content == b""
        assert client.get("/api/tasks").json() == []
        assert client.head("/api/auth/github/start").status_code == 405
        state = start(client)
        callback = "/api/auth/github/callback?state=" + state + "&code=one"
        assert client.head(callback).status_code == 405
        assert calls == []
        assert client.get(callback, follow_redirects=False).headers["location"] == "/"
        assert len(calls) == 2


def test_public_example_links_remain_accessible_without_changing_private_catalog(app, monkeypatch):
    from medsegagent.examples import Examples

    attribution = {"notice_url": "/api/examples/sample/license"}
    monkeypatch.setattr(
        Examples,
        "catalog",
        lambda self: [
            {
                "id": "sample",
                "preview_url": "/api/examples/sample/preview",
                "attribution": attribution,
            }
        ],
    )
    with TestClient(app, base_url="https://testserver") as client:
        public = client.get("/a2a/config").json()["examples"][0]
        assert public["preview_url"] == "/a2a/examples/sample/preview"
        assert public["attribution"]["notice_url"] == "/a2a/examples/sample/license"
        client.post("/api/auth/guest")
        private = client.get("/api/config").json()["examples"][0]
        assert private["preview_url"] == "/api/examples/sample/preview"
        assert private["attribution"]["notice_url"] == "/api/examples/sample/license"
