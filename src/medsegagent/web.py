"""Authenticated same-origin Web API; all computation is delegated to Service."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import shutil
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote, urlsplit

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from medsegagent import agent
from medsegagent.service import Service, ServiceError

STATIC = Path(__file__).parent / "web_static"
COOKIE = "medseg_session"


async def json_body(request, limit=16384):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise ServiceError("REQUEST_TOO_LARGE", "Request body too large.", 413)
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError):
        raise ServiceError("INVALID_PARAMS", "Expected a JSON object.") from None
    if not isinstance(value, dict):
        raise ServiceError("INVALID_PARAMS", "Expected a JSON object.")
    return value


def create_app(root: Path | None = None, public_url: str | None = None, tokens=None):
    public_url = public_url or os.environ.get("MEDSEGAGENT_PUBLIC_URL", "http://127.0.0.1:8767")
    url = urlsplit(public_url)
    if url.scheme != "https" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("A public service requires HTTPS.")
    if url.path.rstrip("/") or url.query or url.fragment or url.username:
        raise ValueError("MEDSEGAGENT_PUBLIC_URL must be an origin without path or credentials.")
    tokens = (
        tokens
        if tokens is not None
        else json.loads(os.environ.get("MEDSEGAGENT_TOKENS_JSON", "{}"))
    )
    if (
        not isinstance(tokens, dict)
        or not tokens
        or any(
            not isinstance(k, str) or not isinstance(v, str) or len(v) < 32
            for k, v in tokens.items()
        )
        or len(set(tokens.values())) != len(tokens)
    ):
        raise ValueError("Configure distinct Bearer tokens of at least 32 characters per identity.")
    service = Service(root or Path(os.environ.get("MEDSEGAGENT_DATA_ROOT", "runtime")), public_url)
    logins = defaultdict(list)

    def token_principal(value):
        for principal, token in tokens.items():
            if hmac.compare_digest(value.encode(), token.encode()):
                return principal
        raise ServiceError("UNAUTHENTICATED", "Authentication required.", 401)

    def same_origin(request):
        origin = request.headers.get("origin")
        if origin and origin != public_url.rstrip("/"):
            raise ServiceError("FORBIDDEN", "Cross-origin request rejected.", 403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise ServiceError("FORBIDDEN", "Cross-site request rejected.", 403)

    def authenticate(request):
        authorization = request.headers.get("authorization")
        if authorization is not None:
            if not authorization.startswith("Bearer "):
                raise ServiceError("UNAUTHENTICATED", "Bearer authentication required.", 401)
            return token_principal(authorization[7:])
        # A2A always uses explicit Bearer, never a browser session.
        if request.url.path.startswith("/a2a/"):
            raise ServiceError("UNAUTHENTICATED", "Bearer authentication required.", 401)
        session = request.cookies.get(COOKIE, "")
        row = service.db.execute(
            "SELECT principal FROM sessions WHERE hash=? AND expires>?",
            (hashlib.sha256(session.encode()).hexdigest(), time.time()),
        ).fetchone()
        if not row or row[0] not in tokens:
            raise ServiceError("UNAUTHENTICATED", "Authentication required.", 401)
        if request.method not in {"GET", "HEAD"}:
            same_origin(request)
        return row[0]

    async def session(request):
        if request.method == "POST":
            same_origin(request)
            peer = request.client.host if request.client else "unknown"
            now = time.monotonic()
            logins[peer] = [t for t in logins[peer] if t > now - 60]
            if len(logins[peer]) >= 10:
                raise ServiceError("RATE_LIMITED", "Too many login attempts; wait one minute.", 429)
            logins[peer].append(now)
            body = await json_body(request, 2048)
            token = body.get("token", "")
            if not isinstance(token, str):
                raise ServiceError("UNAUTHENTICATED", "Invalid token.", 401)
            principal = token_principal(token)
            value = secrets.token_urlsafe(32)
            with service.db:
                service.db.execute(
                    "INSERT INTO sessions VALUES(?,?,?)",
                    (hashlib.sha256(value.encode()).hexdigest(), principal, time.time() + 43200),
                )
            response = JSONResponse({"principal": principal, "authenticated": True})
            response.set_cookie(
                COOKIE,
                value,
                max_age=43200,
                httponly=True,
                secure=url.scheme == "https",
                samesite="strict",
            )
            return response
        principal = authenticate(request)
        if request.method == "DELETE":
            with service.db:
                service.db.execute(
                    "DELETE FROM sessions WHERE hash=?",
                    (hashlib.sha256(request.cookies.get(COOKIE, "").encode()).hexdigest(),),
                )
            response = JSONResponse({"authenticated": False})
            response.delete_cookie(COOKIE)
            return response
        return JSONResponse({"principal": principal, "authenticated": True})

    async def config(request):
        authenticate(request)
        return JSONResponse(
            {
                "max_upload_bytes": service.max_upload_bytes,
                "retention_hours": service.retention_seconds // 3600,
                "tools": list(agent.TOOLS),
                "modalities": ["CT", "MR"],
                "modality_input": "request_text",
                "model": agent.MODEL,
                "warning": "Research use only. No clinical validation.",
            }
        )

    async def upload(request):
        principal = authenticate(request)
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > service.max_upload_bytes):
            raise ServiceError("REQUEST_TOO_LARGE", "Maximum upload is 90 MiB.", 413)
        filename = unquote(request.headers.get("x-filename", "image.nii.gz"))
        upload_id, path = service.reserve_upload(principal, filename)
        success = False
        try:
            size = 0
            async with asyncio.timeout(180):
                with path.open("xb") as stream:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > service.max_upload_bytes:
                            raise ServiceError(
                                "REQUEST_TOO_LARGE", "Maximum upload is 90 MiB.", 413
                            )
                        stream.write(chunk)
                data = await service.finish_upload(principal, upload_id, path, filename, size)
            success = True
            return JSONResponse(data, status_code=201)
        except TimeoutError:
            raise ServiceError("UPLOAD_TIMEOUT", "Upload or validation timed out.", 408) from None
        except ServiceError:
            raise
        except ValueError:
            raise ServiceError(
                "INVALID_FILE", "Invalid, non-finite, oversized or truncated 3D NIfTI."
            ) from None
        finally:
            service.uploading.discard(principal)
            if not success:
                shutil.rmtree(path.parent, ignore_errors=True)

    async def upload_file(request):
        principal = authenticate(request)
        upload_id = request.path_params["upload_id"]
        metadata = service.get_upload(principal, upload_id)
        path = service.upload_path(principal, upload_id)
        return FileResponse(
            path,
            media_type="application/gzip" if path.suffix == ".gz" else "application/octet-stream",
            filename=metadata.get("name", path.name),
        )

    async def upload_detail(request):
        principal = authenticate(request)
        upload_id = request.path_params["upload_id"]
        if request.method == "DELETE":
            service.delete_upload(principal, upload_id)
            return JSONResponse({"deleted": True})
        return JSONResponse(service.get_upload(principal, upload_id))

    async def tasks(request):
        principal = authenticate(request)
        if request.method == "GET":
            return JSONResponse(service.list(principal))
        body = await json_body(request)
        if set(body) - {"upload_id", "text", "modality", "message_id"}:
            raise ServiceError("INVALID_PARAMS", "Unknown task fields.")
        row = await service.submit(
            principal,
            body.get("upload_id"),
            body.get("text"),
            body.get("modality"),
            body.get("message_id"),
        )
        return JSONResponse(row, status_code=202)

    async def task(request):
        return JSONResponse(service.get(authenticate(request), request.path_params["task_id"]))

    async def cancel(request):
        return JSONResponse(
            await service.cancel(authenticate(request), request.path_params["task_id"])
        )

    async def result_file(request):
        path = service.file_path(
            authenticate(request), request.path_params["task_id"], request.path_params["name"]
        )
        return FileResponse(
            path,
            media_type="application/json" if path.suffix == ".json" else "application/gzip",
            filename=path.name,
        )

    async def index(request):
        return FileResponse(STATIC / "index.html")

    async def health(request):
        return JSONResponse({"status": "ok"})

    async def ready(request):
        configured = bool(os.environ.get("OPENAI_API_KEY")) and bool(
            os.environ.get("OPENAI_BASE_URL")
        )
        service.db.execute("SELECT 1").fetchone()
        return JSONResponse(
            {
                "status": "ready" if configured else "not_ready",
                "model_configured": configured,
                "scope": "process, configuration and SQLite; inference not tested",
            },
            status_code=200 if configured else 503,
        )

    async def error(request, exc):
        code = getattr(exc, "code", "INVALID_PARAMS")
        return JSONResponse(
            {"error": {"code": code, "message": str(exc)}},
            status_code=getattr(exc, "status_code", 400),
        )

    @asynccontextmanager
    async def lifespan(app):
        await service.start()
        try:
            yield
        finally:
            await service.close()

    routes = [
        Route("/", index),
        Route("/healthz", health),
        Route("/readyz", ready),
        Route("/api/session", session, methods=["GET", "POST", "DELETE"]),
        Route("/api/config", config),
        Route("/api/uploads", upload, methods=["POST"]),
        Route("/api/uploads/{upload_id}/file", upload_file),
        Route("/api/uploads/{upload_id}", upload_detail, methods=["GET", "DELETE"]),
        Route("/api/tasks", tasks, methods=["GET", "POST"]),
        Route("/api/tasks/{task_id}", task),
        Route("/api/tasks/{task_id}/cancel", cancel, methods=["POST"]),
        Route("/api/tasks/{task_id}/files/{name}", result_file),
        Mount("/static", StaticFiles(directory=STATIC, check_dir=False)),
    ]
    from medsegagent.a2a import routes as a2a_routes

    routes.extend(a2a_routes(service, authenticate))
    app = Starlette(
        routes=routes,
        lifespan=lifespan,
        exception_handlers={ServiceError: error, agent.RoutingError: error},
    )
    app.state.service = service

    async def security_headers(request: Request, call_next):
        # No arbitrary Host values; proxy can connect through loopback while preserving public Host.
        if request.url.hostname not in {url.hostname, "localhost", "127.0.0.1", "testserver"}:
            return JSONResponse({"error": "Invalid Host"}, status_code=400)
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - do not expose paths or secret-bearing tracebacks.
            response = JSONResponse(
                {"error": {"code": "INTERNAL_ERROR", "message": "Internal service error."}},
                status_code=500,
            )
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "X-Frame-Options": "DENY",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' blob: data:; connect-src 'self' blob:; worker-src 'self' blob:; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            }
        )
        if url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    app.add_middleware(BaseHTTPMiddleware, dispatch=security_headers)
    return app
