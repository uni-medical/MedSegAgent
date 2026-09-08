"""Persistent guest and GitHub identities; provider credentials never become sessions.

All database operations are synchronous on the Service event loop. OAuth network
I/O happens only after the one-use transaction has committed. Request adapters
must check the fixed origin for browser mutations; this module never reads Host,
Authorization, or the legacy token-session table to authorize a request.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from urllib.parse import urlencode, urlsplit

import httpx

SESSION_COOKIE = "medseg_session_v2"
OAUTH_COOKIE = "medseg_oauth_github"
SESSION_TTL_SECONDS = 30 * 24 * 3600
OAUTH_TTL_SECONDS = 600
_MAX_RESPONSE_BYTES = 64 * 1024
_MAX_PENDING_LOGINS = 10000
_OPAQUE = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_CODE = re.compile(r"[A-Za-z0-9._~-]{1,2048}\Z")
_VERIFIER = re.compile(r"[A-Za-z0-9._~-]{43,128}\Z")


class AuthError(ValueError):
    """An adapter-safe failure, without credentials or provider response bodies."""

    def __init__(self, code, message, status_code=400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _oauth_error():
    return AuthError("OAUTH_FAILED", "GitHub 登录未完成，请重试。", 401)


def _digest(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def pkce_challenge(verifier):
    if not isinstance(verifier, str) or not _VERIFIER.fullmatch(verifier):
        raise _oauth_error()
    return (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .decode("ascii")
        .rstrip("=")
    )


def _credential(value):
    return (
        isinstance(value, str)
        and 0 < len(value) <= 4096
        and all(32 < ord(char) < 127 for char in value)
    )


def _public_origin(value):
    try:
        if not isinstance(value, str) or any(
            ord(char) <= 32 or ord(char) >= 127 or char in "\\?#" for char in value
        ):
            raise ValueError
        parsed = urlsplit(value)
        if (
            not parsed.hostname
            or "@" in parsed.netloc
            or parsed.path not in {"", "/"}
            or parsed.netloc.endswith(":")
            or parsed.port is not None
            and not 1 <= parsed.port <= 65535
            or not (
                parsed.scheme == "https"
                or parsed.scheme == "http"
                and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            )
        ):
            raise ValueError
    except ValueError:
        raise ValueError(
            "MEDSEGAGENT_PUBLIC_URL must be a fixed HTTPS origin (HTTP only on loopback)."
        ) from None
    return value.rstrip("/")


@dataclass(frozen=True, slots=True)
class Principal:
    id: str
    kind: str
    name: str

    def as_dict(self):
        return {"id": self.id, "kind": self.kind, "name": self.name}


@dataclass(frozen=True, slots=True)
class IssuedSession:
    token: str = field(repr=False)
    principal: Principal
    expires_at: float


@dataclass(frozen=True, slots=True)
class OAuthStart:
    url: str
    browser_nonce: str = field(repr=False)
    expires_at: float


class AuthStore:
    def __init__(
        self, db, public_url, *, github_client_id=None, github_client_secret=None, transport=None
    ):
        self.db = db
        self.public_url = _public_origin(public_url)
        self.cookie_secure = urlsplit(self.public_url).scheme == "https"
        self.callback_url = self.public_url + "/api/auth/github/callback"
        self._client_id = (
            os.environ.get("MEDSEGAGENT_GITHUB_CLIENT_ID", "").strip()
            if github_client_id is None
            else github_client_id
        )
        self._client_secret = (
            os.environ.get("MEDSEGAGENT_GITHUB_CLIENT_SECRET", "").strip()
            if github_client_secret is None
            else github_client_secret
        )
        if (self._client_id or self._client_secret) and not (
            _credential(self._client_id) and _credential(self._client_secret)
        ):
            raise ValueError(
                "MEDSEGAGENT_GITHUB_CLIENT_ID and CLIENT_SECRET must be configured together."
            )
        self.github_enabled = bool(self._client_id)
        self._transport = transport
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS auth_principals (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK(kind IN ('guest','github')),
                name TEXT NOT NULL,
                created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS auth_sessions (
                hash TEXT PRIMARY KEY,
                principal TEXT NOT NULL REFERENCES auth_principals(id),
                expires REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS auth_sessions_expiry ON auth_sessions(expires);
            CREATE INDEX IF NOT EXISTS auth_sessions_principal ON auth_sessions(principal);
            CREATE TABLE IF NOT EXISTS auth_oauth_transactions (
                state_hash TEXT PRIMARY KEY,
                browser_hash TEXT NOT NULL,
                code_verifier TEXT NOT NULL,
                client_id TEXT NOT NULL,
                callback_url TEXT NOT NULL,
                expires REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS auth_oauth_expiry ON auth_oauth_transactions(expires);
        """)
        self.cleanup()

    def cleanup(self):
        """Expire credentials and orphan guests without changing resource ownership."""
        now = time.time()
        tables = {
            row[0]
            for row in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('tasks','uploads','upload_sessions')"
            )
        }
        owners = " UNION ".join(
            f"SELECT principal FROM {table} WHERE principal IS NOT NULL" for table in sorted(tables)
        )
        preserve_owners = f" AND id NOT IN ({owners})" if owners else ""
        with self.db:
            self.db.execute("DELETE FROM auth_sessions WHERE expires<=?", (now,))
            self.db.execute("DELETE FROM auth_oauth_transactions WHERE expires<=?", (now,))
            self.db.execute(
                "DELETE FROM auth_principals WHERE kind='guest' AND NOT EXISTS "
                "(SELECT 1 FROM auth_sessions WHERE principal=auth_principals.id)" + preserve_owners
            )
        self._next_cleanup = now + 60

    def _maybe_cleanup(self):
        if time.time() >= self._next_cleanup:
            self.cleanup()

    def authenticate(self, token):
        self._maybe_cleanup()
        if not isinstance(token, str) or not _OPAQUE.fullmatch(token):
            return None
        row = self.db.execute(
            "SELECT p.id,p.kind,p.name FROM auth_sessions s "
            "JOIN auth_principals p ON p.id=s.principal WHERE s.hash=? AND s.expires>?",
            (_digest(token), time.time()),
        ).fetchone()
        return Principal(*row) if row is not None else None

    def _check_legacy_collision(self, principal_id):
        # Token account names were arbitrary. Never let a newly verified GitHub
        # subject accidentally claim old records named, for example, github:123.
        for table in ("tasks", "uploads", "upload_sessions", "sessions"):
            if (
                self.db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                ).fetchone()
                and self.db.execute(
                    f"SELECT 1 FROM {table} WHERE principal=? LIMIT 1", (principal_id,)
                ).fetchone()
            ):
                raise AuthError("IDENTITY_CONFLICT", "此账号暂不可用，请联系网站维护者。", 409)

    def _issue(self, principal):
        token = secrets.token_urlsafe(32)
        expires = time.time() + SESSION_TTL_SECONDS
        self.db.execute(
            "INSERT INTO auth_sessions(hash,principal,expires) VALUES(?,?,?)",
            (_digest(token), principal.id, expires),
        )
        return IssuedSession(token, principal, expires)

    def create_guest(self):
        self._maybe_cleanup()
        principal = Principal("guest:" + secrets.token_hex(16), "guest", "游客")
        with self.db:
            self._check_legacy_collision(principal.id)
            self.db.execute(
                "INSERT INTO auth_principals(id,kind,name,created) VALUES(?,?,?,?)",
                (principal.id, principal.kind, principal.name, time.time()),
            )
            return self._issue(principal)

    def revoke(self, token):
        if isinstance(token, str) and _OPAQUE.fullmatch(token):
            with self.db:
                self.db.execute("DELETE FROM auth_sessions WHERE hash=?", (_digest(token),))

    def _require_github(self):
        if not self.github_enabled:
            raise AuthError("AUTH_UNAVAILABLE", "GitHub 登录暂未开放。", 404)

    def begin_github(self):
        self._require_github()
        self._maybe_cleanup()
        state, browser_nonce, verifier = (
            secrets.token_urlsafe(32),
            secrets.token_urlsafe(32),
            secrets.token_urlsafe(48),
        )
        expires = time.time() + OAUTH_TTL_SECONDS
        with self.db:
            self.db.execute("DELETE FROM auth_oauth_transactions WHERE expires<=?", (time.time(),))
            if (
                self.db.execute("SELECT COUNT(*) FROM auth_oauth_transactions").fetchone()[0]
                >= _MAX_PENDING_LOGINS
            ):
                raise AuthError("AUTH_BUSY", "登录请求较多，请稍后重试。", 429)
            self.db.execute(
                "INSERT INTO auth_oauth_transactions VALUES(?,?,?,?,?,?)",
                (
                    _digest(state),
                    _digest(browser_nonce),
                    verifier,
                    self._client_id,
                    self.callback_url,
                    expires,
                ),
            )
        url = "https://github.com/login/oauth/authorize?" + urlencode(
            {
                "client_id": self._client_id,
                "redirect_uri": self.callback_url,
                "scope": "",
                "state": state,
                "code_challenge": pkce_challenge(verifier),
                "code_challenge_method": "S256",
            }
        )
        return OAuthStart(url, browser_nonce, expires)

    async def finish_github(self, *, state, code, browser_nonce):
        self._require_github()
        self._maybe_cleanup()
        if any(
            not isinstance(value, str) or not _OPAQUE.fullmatch(value)
            for value in (state, browser_nonce)
        ):
            raise _oauth_error()
        with self.db:
            row = self.db.execute(
                "DELETE FROM auth_oauth_transactions WHERE state_hash=? AND browser_hash=? "
                "AND expires>? AND client_id=? AND callback_url=? RETURNING code_verifier",
                (
                    _digest(state),
                    _digest(browser_nonce),
                    time.time(),
                    self._client_id,
                    self.callback_url,
                ),
            ).fetchone()
        # Consume a valid attempt even for denied authorization or provider failure.
        if row is None or not isinstance(code, str) or not _CODE.fullmatch(code):
            raise _oauth_error()
        verifier = row[0]
        async with httpx.AsyncClient(
            timeout=10, follow_redirects=False, trust_env=False, transport=self._transport
        ) as client:
            token = await self._json(
                client,
                "POST",
                "https://github.com/login/oauth/access_token",
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "code": code,
                    "redirect_uri": self.callback_url,
                    "code_verifier": verifier,
                },
            )
            access_token = token.get("access_token")
            if (
                "error" in token
                or not _credential(access_token)
                or not isinstance(token.get("token_type"), str)
                or token["token_type"].lower() != "bearer"
            ):
                raise _oauth_error()
            user = await self._json(
                client,
                "GET",
                "https://api.github.com/user",
                headers={"Authorization": "Bearer " + access_token},
            )
        subject, login, name = user.get("id"), user.get("login"), user.get("name")
        if (
            type(subject) is not int
            or subject < 1
            or subject >= 2**63
            or not isinstance(login, str)
            or not re.fullmatch(r"[A-Za-z0-9-]{1,39}", login)
            or name is not None
            and not isinstance(name, str)
        ):
            raise _oauth_error()
        display_name = (
            " ".join(
                "".join(
                    char for char in (name or login) if ord(char) >= 32 and ord(char) != 127
                ).split()
            )[:80]
            or login
        )
        principal = Principal(f"github:{subject}", "github", display_name)
        with self.db:
            existing = self.db.execute(
                "SELECT kind FROM auth_principals WHERE id=?", (principal.id,)
            ).fetchone()
            if existing is None:
                self._check_legacy_collision(principal.id)
                self.db.execute(
                    "INSERT INTO auth_principals(id,kind,name,created) VALUES(?,?,?,?)",
                    (principal.id, principal.kind, principal.name, time.time()),
                )
            elif existing[0] != "github":
                raise _oauth_error()
            else:
                self.db.execute(
                    "UPDATE auth_principals SET name=? WHERE id=?", (principal.name, principal.id)
                )
            return self._issue(principal)

    @staticmethod
    async def _json(client, method, url, **kwargs):
        headers = {
            "Accept": "application/json",
            "User-Agent": "MedSegAgent-Login",
            **kwargs.pop("headers", {}),
        }
        try:
            async with client.stream(method, url, headers=headers, **kwargs) as response:
                if response.status_code != 200:
                    raise _oauth_error()
                length = response.headers.get("Content-Length")
                if length is not None and not 0 <= int(length) <= _MAX_RESPONSE_BYTES:
                    raise _oauth_error()
                body = bytearray()
                async for block in response.aiter_bytes():
                    body.extend(block)
                    if len(body) > _MAX_RESPONSE_BYTES:
                        raise _oauth_error()
                value = json.loads(body)
                if not isinstance(value, dict):
                    raise _oauth_error()
                return value
        except (httpx.HTTPError, ValueError, TypeError):
            raise _oauth_error() from None

    def set_session_cookie(self, response, issued):
        response.set_cookie(
            SESSION_COOKIE,
            issued.token,
            max_age=SESSION_TTL_SECONDS,
            httponly=True,
            secure=self.cookie_secure,
            samesite="lax",
            path="/",
        )
        return response

    def clear_session_cookie(self, response):
        response.delete_cookie(
            SESSION_COOKIE, httponly=True, secure=self.cookie_secure, samesite="lax", path="/"
        )
        return response

    def set_oauth_cookie(self, response, started):
        response.set_cookie(
            OAUTH_COOKIE,
            started.browser_nonce,
            max_age=OAUTH_TTL_SECONDS,
            httponly=True,
            secure=self.cookie_secure,
            samesite="lax",
            path="/api/auth/",
        )
        return response

    def clear_oauth_cookie(self, response):
        response.delete_cookie(
            OAUTH_COOKIE,
            httponly=True,
            secure=self.cookie_secure,
            samesite="lax",
            path="/api/auth/",
        )
        return response
