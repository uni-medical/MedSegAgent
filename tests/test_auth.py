"""Offline identity, session and browser-bound OAuth acceptance checks."""

import asyncio
import hashlib
import sqlite3
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from starlette.responses import Response

from medsegagent import auth
from medsegagent.auth import OAUTH_COOKIE, SESSION_COOKIE, AuthError, AuthStore


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("MEDSEGAGENT_GITHUB_CLIENT_ID", raising=False)
    monkeypatch.delenv("MEDSEGAGENT_GITHUB_CLIENT_SECRET", raising=False)
    connection = sqlite3.connect(tmp_path / "state.sqlite3")
    connection.row_factory = sqlite3.Row
    yield connection
    connection.close()


def store(db, handler=None, **kwargs):
    return AuthStore(
        db,
        kwargs.pop("public_url", "https://medseg.example"),
        github_client_id=kwargs.pop("github_client_id", "test-client"),
        github_client_secret=kwargs.pop("github_client_secret", "synthetic-client-secret"),
        transport=httpx.MockTransport(handler) if handler is not None else None,
        **kwargs,
    )


def provider(requests, *, subject=123, login="researcher", name="Research User"):
    def handle(request):
        requests.append(request)
        if request.url == "https://github.com/login/oauth/access_token":
            return httpx.Response(
                200, json={"token_type": "bearer", "access_token": "private-provider-token"}
            )
        assert request.url == "https://api.github.com/user"
        assert request.headers["Authorization"] == "Bearer private-provider-token"
        return httpx.Response(200, json={"id": subject, "login": login, "name": name})

    return handle


def begin(instance):
    started = instance.begin_github()
    state = parse_qs(urlsplit(started.url).query)["state"][0]
    return started, state


def finish(instance, started, state, code="authorization-code"):
    return asyncio.run(
        instance.finish_github(state=state, code=code, browser_nonce=started.browser_nonce)
    )


def test_guest_identities_and_hashed_sessions_survive_reopening(db):
    instance = AuthStore(db, "http://127.0.0.1:8767")
    assert not instance.github_enabled
    first, second = instance.create_guest(), instance.create_guest()
    assert first.principal.id != second.principal.id
    assert first.principal.id.startswith("guest:") and first.principal.kind == "guest"
    assert first.principal.as_dict() == {"id": first.principal.id, "kind": "guest", "name": "游客"}
    assert instance.authenticate(first.token) == first.principal
    persisted = "\n".join(db.iterdump())
    assert first.token not in persisted and second.token not in persisted
    assert hashlib.sha256(first.token.encode()).hexdigest() in persisted
    assert first.token not in repr(first)
    restarted = AuthStore(db, "http://127.0.0.1:8767")
    assert restarted.authenticate(first.token) == first.principal
    assert restarted.authenticate(second.token) == second.principal
    assert "quota" not in persisted and "daily" not in persisted


def test_revoke_and_expiry_remove_only_unreachable_guest_identities(db, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(auth.time, "time", lambda: clock[0])
    instance = store(db)
    first, second = instance.create_guest(), instance.create_guest()
    instance.revoke(first.token)
    assert instance.authenticate(first.token) is None
    assert instance.authenticate(second.token) == second.principal
    clock[0] = second.expires_at
    assert instance.authenticate(second.token) is None
    instance.cleanup()
    assert db.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM auth_principals").fetchone()[0] == 0


def test_opportunistic_cleanup_keeps_github_identity_and_all_research_records(db, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(auth.time, "time", lambda: clock[0])
    instance = store(db, provider([]))
    guest = instance.create_guest()
    started, state = begin(instance)
    github = finish(instance, started, state)
    db.execute("CREATE TABLE tasks(principal TEXT, data TEXT)")
    db.executemany(
        "INSERT INTO tasks VALUES(?,?)",
        [(guest.principal.id, "guest-record"), (github.principal.id, "github-record")],
    )
    db.commit()
    clock[0] = github.expires_at
    assert instance.authenticate(None) is None
    assert db.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0
    assert {row[0] for row in db.execute("SELECT id FROM auth_principals")} == {
        guest.principal.id,
        github.principal.id,
    }
    assert db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 2
    started, state = begin(instance)
    assert finish(instance, started, state).principal.id == github.principal.id


@pytest.mark.parametrize("table", ["tasks", "uploads", "upload_sessions"])
def test_cleanup_keeps_guest_resource_ownership_until_the_last_resource_is_removed(db, table):
    instance = store(db)
    guest = instance.create_guest()
    db.execute(f"CREATE TABLE {table}(principal TEXT)")
    db.execute(f"INSERT INTO {table} VALUES(?)", (guest.principal.id,))
    db.commit()
    instance.revoke(guest.token)
    instance.cleanup()
    assert db.execute("SELECT id FROM auth_principals").fetchone()[0] == guest.principal.id
    db.execute(f"DELETE FROM {table}")
    db.commit()
    instance.cleanup()
    assert db.execute("SELECT COUNT(*) FROM auth_principals").fetchone()[0] == 0


def test_pending_login_capacity_recovers_after_expiry_without_daily_quota(db, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(auth.time, "time", lambda: clock[0])
    monkeypatch.setattr(auth, "_MAX_PENDING_LOGINS", 2)
    instance = store(db)
    begin(instance)
    begin(instance)
    with pytest.raises(AuthError) as error:
        begin(instance)
    assert error.value.code == "AUTH_BUSY" and error.value.status_code == 429
    clock[0] += auth.OAUTH_TTL_SECONDS
    begin(instance)
    assert db.execute("SELECT COUNT(*) FROM auth_oauth_transactions").fetchone()[0] == 1


@pytest.mark.parametrize("token", [None, "", "a" * 42, "a" * 44, "a" * 100000, "中" * 43, [], {}])
def test_invalid_session_values_never_authorize(db, token):
    instance = store(db)
    assert instance.authenticate(token) is None
    instance.revoke(token)


def test_old_token_admin_sessions_are_never_authorized_or_modified(db):
    token = "a" * 43
    db.execute("CREATE TABLE sessions(hash TEXT, principal TEXT, expires REAL)")
    db.execute(
        "INSERT INTO sessions VALUES(?,?,?)",
        (hashlib.sha256(token.encode()).hexdigest(), "admin", 9e12),
    )
    db.commit()
    original = tuple(db.execute("SELECT * FROM sessions").fetchone())
    instance = store(db)
    assert instance.authenticate(token) is None
    guest = instance.create_guest()
    assert guest.principal.id != "admin"
    assert tuple(db.execute("SELECT * FROM sessions").fetchone()) == original
    assert SESSION_COOKIE != "medseg_session"


@pytest.mark.parametrize(
    "origin",
    [
        "http://example.com",
        "ftp://localhost",
        "https://example.com/path",
        "https://u:p@example.com",
        "https://example.com?",
        "https://example.com#",
        "https://example.com\\bad",
        "https://example.com:0",
        "https://example.com:99999",
        "https://example.com:",
        " https://example.com",
        "https://",
        None,
    ],
)
def test_public_origin_rejects_untrusted_or_malformed_callbacks(db, origin):
    with pytest.raises(ValueError, match="fixed HTTPS origin"):
        store(db, public_url=origin)


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:8767",
        "http://127.0.0.1:8767/",
        "http://[::1]:8767",
        "https://medseg.example/",
    ],
)
def test_fixed_origin_allows_https_and_loopback_only(db, origin):
    instance = store(db, public_url=origin)
    assert instance.callback_url == origin.rstrip("/") + "/api/auth/github/callback"
    assert instance.cookie_secure == origin.startswith("https:")


def test_credentials_are_optional_paired_and_loaded_only_from_new_environment(db, monkeypatch):
    monkeypatch.setenv("MEDSEGAGENT_TOKENS_JSON", '{"admin":"old-token"}')
    assert not AuthStore(db, "https://medseg.example").github_enabled
    monkeypatch.setenv("MEDSEGAGENT_GITHUB_CLIENT_ID", "new-client")
    with pytest.raises(ValueError, match="configured together"):
        AuthStore(db, "https://medseg.example")
    monkeypatch.setenv("MEDSEGAGENT_GITHUB_CLIENT_SECRET", "new-secret")
    instance = AuthStore(db, "https://medseg.example")
    assert instance.github_enabled
    assert "new-secret" not in repr(instance)
    with pytest.raises(ValueError, match="configured together"):
        store(db, github_client_secret="header\ninjection")


def test_disabled_github_has_no_login_transaction(db):
    instance = AuthStore(db, "https://medseg.example")
    with pytest.raises(AuthError) as error:
        instance.begin_github()
    assert error.value.code == "AUTH_UNAVAILABLE"
    assert db.execute("SELECT COUNT(*) FROM auth_oauth_transactions").fetchone()[0] == 0


def test_pkce_matches_rfc7636_vector():
    assert (
        auth.pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk")
        == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    )


def test_authorization_requests_no_scopes_and_binds_state_to_browser_and_configuration(db):
    instance = store(db)
    started, state = begin(instance)
    query = parse_qs(urlsplit(started.url).query, keep_blank_values=True)
    assert urlsplit(started.url).netloc == "github.com"
    assert query["redirect_uri"] == [instance.callback_url]
    assert query["scope"] == [""]
    assert query["code_challenge_method"] == ["S256"]
    row = db.execute("SELECT * FROM auth_oauth_transactions").fetchone()
    assert query["code_challenge"] == [auth.pkce_challenge(row["code_verifier"])]
    assert row["state_hash"] == hashlib.sha256(state.encode()).hexdigest()
    assert row["browser_hash"] == hashlib.sha256(started.browser_nonce.encode()).hexdigest()
    assert row["client_id"] == "test-client" and row["callback_url"] == instance.callback_url
    persisted = "\n".join(db.iterdump())
    assert state not in persisted and started.browser_nonce not in persisted
    assert started.browser_nonce not in repr(started)


def test_github_uses_stable_numeric_identity_and_does_not_persist_provider_tokens(db):
    requests = []
    instance = store(db, provider(requests))
    guest = instance.create_guest()
    started, state = begin(instance)
    transaction = db.execute("SELECT code_verifier FROM auth_oauth_transactions").fetchone()[0]
    issued = finish(instance, started, state)
    assert issued.principal.as_dict() == {
        "id": "github:123",
        "kind": "github",
        "name": "Research User",
    }
    form = parse_qs(requests[0].content.decode())
    assert form == {
        "client_id": ["test-client"],
        "client_secret": ["synthetic-client-secret"],
        "code": ["authorization-code"],
        "redirect_uri": [instance.callback_url],
        "code_verifier": [transaction],
    }
    assert instance.authenticate(issued.token) == issued.principal
    assert instance.authenticate(guest.token) == guest.principal
    instance = store(db, provider(requests, login="renamed", name="New Display Name"))
    second_start, second_state = begin(instance)
    second = finish(instance, second_start, second_state)
    assert second.principal.id == issued.principal.id and second.token != issued.token
    assert second.principal.name == "New Display Name"
    assert db.execute("SELECT COUNT(*) FROM auth_principals").fetchone()[0] == 2
    assert db.execute("SELECT COUNT(*) FROM auth_oauth_transactions").fetchone()[0] == 0
    persisted = "\n".join(db.iterdump())
    for secret in (
        "private-provider-token",
        "synthetic-client-secret",
        "authorization-code",
        transaction,
        issued.token,
    ):
        assert secret not in persisted


def test_wrong_browser_cannot_consume_state_and_success_cannot_be_replayed(db):
    requests = []
    instance = store(db, provider(requests))
    started, state = begin(instance)
    with pytest.raises(AuthError):
        asyncio.run(instance.finish_github(state=state, code="code", browser_nonce="x" * 43))
    assert requests == []
    assert finish(instance, started, state).principal.id == "github:123"
    with pytest.raises(AuthError):
        finish(instance, started, state)
    assert len(requests) == 2


@pytest.mark.parametrize("change", ["client", "origin", "expiry"])
def test_pending_state_cannot_cross_client_origin_or_expiry(db, monkeypatch, change):
    requests = []
    instance = store(db, provider(requests))
    started, state = begin(instance)
    kwargs = {"github_client_id": "different-client"} if change == "client" else {}
    if change == "origin":
        kwargs["public_url"] = "https://different.example"
    if change == "expiry":
        monkeypatch.setattr(auth.time, "time", lambda: started.expires_at)
    reopened = store(db, provider(requests), **kwargs)
    with pytest.raises(AuthError):
        finish(reopened, started, state)
    assert requests == []


def test_pending_login_survives_restart_with_the_same_configuration(db):
    started, state = begin(store(db))
    requests = []
    restarted = store(db, provider(requests))
    assert finish(restarted, started, state).principal.id == "github:123"


@pytest.mark.parametrize("code", ["", None, "bad\ncode", "x" * 2049])
def test_denied_or_invalid_code_consumes_the_bound_transaction(db, code):
    requests = []
    instance = store(db, provider(requests))
    started, state = begin(instance)
    with pytest.raises(AuthError):
        finish(instance, started, state, code=code)
    with pytest.raises(AuthError):
        finish(instance, started, state)
    assert requests == []


def test_concurrent_callbacks_exchange_one_authorization_code_only_once(db):
    requests = []
    handler = provider(requests)

    async def slow(request):
        await asyncio.sleep(0)
        return handler(request)

    instance = store(db, slow)
    started, state = begin(instance)

    async def scenario():
        return await asyncio.gather(
            *[
                instance.finish_github(
                    state=state, code="code", browser_nonce=started.browser_nonce
                )
                for _ in range(2)
            ],
            return_exceptions=True,
        )

    replies = asyncio.run(scenario())
    assert sum(isinstance(reply, auth.IssuedSession) for reply in replies) == 1
    assert sum(isinstance(reply, AuthError) for reply in replies) == 1
    assert len(requests) == 2


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(302, headers={"Location": "https://attacker.example"}),
        httpx.Response(500, text="private-provider-token"),
        httpx.Response(200, json={"error": "private-provider-token"}),
        httpx.Response(200, json={"token_type": "mac", "access_token": "secret"}),
        httpx.Response(200, json={"token_type": "bearer", "access_token": "bad\nheader"}),
        httpx.Response(200, json=[]),
        httpx.Response(200, content=b"not JSON"),
        httpx.Response(200, content=b"x" * (64 * 1024 + 1)),
        httpx.Response(200, headers={"Content-Length": "bogus"}, content=b"{}"),
    ],
)
def test_provider_failures_are_bounded_safe_and_not_retried(db, response):
    requests = []

    def handle(request):
        requests.append(request)
        return response

    instance = store(db, handle)
    started, state = begin(instance)
    with pytest.raises(AuthError) as error:
        finish(instance, started, state)
    assert "private-provider-token" not in str(error.value)
    assert len(requests) == 1
    with pytest.raises(AuthError):
        finish(instance, started, state)
    assert len(requests) == 1
    assert db.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0


@pytest.mark.parametrize(
    "subject,login,name",
    [
        (True, "valid", None),
        (0, "valid", None),
        ("123", "valid", None),
        (123, "", None),
        (123, "valid", {}),
        (2**63, "valid", None),
    ],
)
def test_invalid_github_identity_never_creates_an_account(db, subject, login, name):
    instance = store(db, provider([], subject=subject, login=login, name=name))
    started, state = begin(instance)
    with pytest.raises(AuthError):
        finish(instance, started, state)
    assert db.execute("SELECT COUNT(*) FROM auth_principals").fetchone()[0] == 0


def test_network_errors_do_not_expose_provider_request_details(db):
    def fail(request):
        raise httpx.ConnectError("synthetic-client-secret private-provider-token", request=request)

    instance = store(db, fail)
    started, state = begin(instance)
    with pytest.raises(AuthError) as error:
        finish(instance, started, state)
    assert str(error.value) == "GitHub 登录未完成，请重试。"


@pytest.mark.parametrize("table", ["sessions", "tasks", "uploads", "upload_sessions"])
def test_new_github_identity_cannot_claim_legacy_token_records_with_the_same_name(db, table):
    db.execute(f"CREATE TABLE {table}(principal TEXT)")
    db.execute(f"INSERT INTO {table} VALUES(?)", ("github:123",))
    db.commit()
    instance = store(db, provider([]))
    started, state = begin(instance)
    with pytest.raises(AuthError) as error:
        finish(instance, started, state)
    assert error.value.code == "IDENTITY_CONFLICT"
    assert db.execute("SELECT COUNT(*) FROM auth_principals").fetchone()[0] == 0
    assert db.execute(f"SELECT principal FROM {table}").fetchone()[0] == "github:123"


@pytest.mark.parametrize(
    "origin,secure", [("https://medseg.example", True), ("http://localhost:8767", False)]
)
def test_cookie_helpers_keep_credentials_opaque_and_http_only(db, origin, secure):
    instance = store(db, public_url=origin)
    issued = instance.create_guest()
    started, _state = begin(instance)
    response = instance.set_session_cookie(Response(), issued)
    cookie = response.headers["set-cookie"]
    assert f"{SESSION_COOKIE}={issued.token}" in cookie
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie and "Path=/" in cookie
    assert "Domain=" not in cookie and ("Secure" in cookie) == secure
    assert f"Max-Age={auth.SESSION_TTL_SECONDS}" in cookie
    assert issued.principal.id not in cookie
    oauth = instance.set_oauth_cookie(Response(), started).headers["set-cookie"]
    assert OAUTH_COOKIE in oauth and "Path=/api/auth/" in oauth and "HttpOnly" in oauth
    assert "SameSite=lax" in oauth and ("Secure" in oauth) == secure
    for method in (instance.clear_session_cookie, instance.clear_oauth_cookie):
        cleared = method(Response()).headers["set-cookie"]
        assert "Max-Age=0" in cleared and "HttpOnly" in cleared
