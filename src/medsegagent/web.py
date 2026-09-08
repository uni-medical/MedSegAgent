"""Guest/GitHub Web access and public A2A; computation is delegated to Service."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote, urlsplit

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from medsegagent import agent
from medsegagent.auth import OAUTH_COOKIE, SESSION_COOKIE, AuthError, AuthStore
from medsegagent.examples import Examples
from medsegagent.service import PUBLIC_A2A_PRINCIPAL, SINGLE_UPLOAD_BYTES, Service, ServiceError
from medsegagent.task_specs import CAPABILITIES
from medsegagent.upload_sessions import CHUNK_BYTES, UploadSessions

STATIC = Path(__file__).parent / "web_static"
COOKIE = SESSION_COOKIE


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


def create_app(root: Path | None = None, public_url: str | None = None):
    public_url = public_url or os.environ.get("MEDSEGAGENT_PUBLIC_URL", "http://127.0.0.1:8767")
    url = urlsplit(public_url)
    if url.scheme != "https" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("A public service requires HTTPS.")
    if url.path.rstrip("/") or url.query or url.fragment or url.username:
        raise ValueError("MEDSEGAGENT_PUBLIC_URL must be an origin without path or credentials.")
    service = Service(root or Path(os.environ.get("MEDSEGAGENT_DATA_ROOT", "runtime")), public_url)
    auth = AuthStore(service.db, public_url)
    public_url = auth.public_url
    examples = Examples(service)
    upload_sessions = UploadSessions(service)
    login_attempts = deque()

    def login_capacity():
        # A shared short burst limit works behind proxies without trusting IP headers.
        now = time.monotonic()
        while login_attempts and login_attempts[0] <= now - 60:
            login_attempts.popleft()
        if len(login_attempts) >= 120:
            raise ServiceError("RATE_LIMITED", "Login is busy; retry in one minute.", 429)
        login_attempts.append(now)

    def same_origin(request):
        origin = request.headers.get("origin")
        if origin and origin != public_url.rstrip("/"):
            raise ServiceError("FORBIDDEN", "Cross-origin request rejected.", 403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise ServiceError("FORBIDDEN", "Cross-site request rejected.", 403)

    def optional_identity(request):
        if request.method not in {"GET", "HEAD"}:
            same_origin(request)
        principal = auth.authenticate(request.cookies.get(COOKIE))
        return principal.id if principal else None

    def authenticate(request):
        principal = optional_identity(request)
        if request.url.path.startswith("/a2a/"):
            return principal or PUBLIC_A2A_PRINCIPAL
        if principal is None:
            raise ServiceError("UNAUTHENTICATED", "Choose guest access or GitHub login.", 401)
        return principal

    def session_data(principal=None):
        return {
            "authenticated": principal is not None,
            "github_enabled": auth.github_enabled,
            "identity": {"kind": principal.kind, "display_name": principal.name}
            if principal
            else None,
        }

    async def session(request):
        if request.method == "DELETE":
            same_origin(request)
            auth.revoke(request.cookies.get(COOKIE))
            return auth.clear_session_cookie(JSONResponse(session_data()))
        return JSONResponse(session_data(auth.authenticate(request.cookies.get(COOKIE))))

    async def guest_login(request):
        same_origin(request)
        current = auth.authenticate(request.cookies.get(COOKIE))
        if current and current.kind == "guest":
            return JSONResponse(session_data(current))
        login_capacity()
        issued = auth.create_guest()
        auth.revoke(request.cookies.get(COOKIE))
        return auth.set_session_cookie(JSONResponse(session_data(issued.principal)), issued)

    async def github_start(request):
        if request.method != "GET":
            return JSONResponse(
                {"error": "Method not allowed"}, status_code=405, headers={"Allow": "GET"}
            )
        same_origin(request)
        login_capacity()
        started = auth.begin_github()
        return auth.set_oauth_cookie(RedirectResponse(started.url, status_code=303), started)

    async def github_callback(request):
        if request.method != "GET":
            return JSONResponse(
                {"error": "Method not allowed"}, status_code=405, headers={"Allow": "GET"}
            )
        try:
            params = request.query_params
            if len(params.getlist("state")) != 1 or len(params.getlist("code")) > 1:
                raise AuthError("OAUTH_FAILED", "GitHub login failed.", 401)
            issued = await auth.finish_github(
                state=params.get("state"),
                code=None if "error" in params else params.get("code"),
                browser_nonce=request.cookies.get(OAUTH_COOKIE),
            )
        except AuthError:
            return auth.clear_oauth_cookie(RedirectResponse("/?error=oauth", status_code=303))
        auth.revoke(request.cookies.get(COOKIE))
        response = auth.set_session_cookie(RedirectResponse("/", status_code=303), issued)
        return auth.clear_oauth_cookie(response)

    async def config(request):
        authenticate(request)
        example_catalog = examples.catalog()
        if request.url.path.startswith("/a2a/"):
            for entry in example_catalog:
                entry["preview_url"] = entry["preview_url"].replace(
                    "/api/examples/", "/a2a/examples/", 1
                )
                attribution = entry.get("attribution", {})
                notice = attribution.get("notice_url", "")
                if notice.startswith("/api/examples/"):
                    entry["attribution"] = {
                        **attribution,
                        "notice_url": notice.replace("/api/examples/", "/a2a/examples/", 1),
                    }
        return JSONResponse(
            {
                "max_upload_bytes": service.max_upload_bytes,
                "single_upload_bytes": SINGLE_UPLOAD_BYTES,
                "upload_chunk_bytes": CHUNK_BYTES,
                "retention_hours": service.retention_seconds // 3600,
                "modalities": ["CT", "MR"],
                "modality_input": "optional_with_detection",
                "capabilities": CAPABILITIES,
                "examples": example_catalog,
                "warning": "Research use only. No clinical validation.",
            }
        )

    async def example_open(request):
        principal = authenticate(request)
        return JSONResponse(
            await examples.open(principal, request.path_params["example_id"]), status_code=201
        )

    async def example_preview(request):
        authenticate(request)
        return FileResponse(
            examples.path(request.path_params["example_id"], "preview"), media_type="image/png"
        )

    async def example_license(request):
        authenticate(request)
        example_id = request.path_params["example_id"]
        return FileResponse(
            examples.path(example_id, "license_file"),
            media_type="text/plain; charset=utf-8",
            filename=example_id + "-LICENSE.txt",
        )

    async def upload(request):
        principal = authenticate(request)
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > SINGLE_UPLOAD_BYTES):
            raise ServiceError(
                "REQUEST_TOO_LARGE", "Use upload sessions for files over 90 MiB.", 413
            )
        filename = unquote(request.headers.get("x-filename", "image.nii.gz"))
        upload_id, path = service.reserve_upload(
            principal, filename, int(length) if length else SINGLE_UPLOAD_BYTES
        )
        success = False
        try:
            size = 0
            async with asyncio.timeout(180):
                with path.open("xb") as stream:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > SINGLE_UPLOAD_BYTES:
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

    async def upload_session_start(request):
        principal = authenticate(request)
        return JSONResponse(
            upload_sessions.start(principal, await json_body(request, 2048)), status_code=201
        )

    async def upload_session(request):
        principal = authenticate(request)
        upload_id = request.path_params["upload_id"]
        if request.method == "PUT":
            return JSONResponse(await upload_sessions.put(principal, upload_id, request))
        if request.method == "DELETE":
            return JSONResponse(upload_sessions.delete(principal, upload_id))
        return JSONResponse(upload_sessions.get(principal, upload_id))

    async def upload_session_complete(request):
        return JSONResponse(
            await upload_sessions.complete(authenticate(request), request.path_params["upload_id"])
        )

    async def upload_file(request):
        principal = authenticate(request)
        upload_id = request.path_params["upload_id"]
        metadata = service.get_upload(principal, upload_id)
        path = service.upload_path(principal, upload_id)
        return FileResponse(
            path,
            media_type="application/gzip" if path.suffix == ".gz" else "application/octet-stream",
            filename=metadata.get("name", path.name),
            headers={
                "Link": f'<{public_url}/{"a2a" if request.url.path.startswith("/a2a/") else "api"}/examples/{metadata["example_id"]}/license>; rel="license"'
            }
            if metadata.get("example_id")
            else None,
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
        if request.method in {"GET", "HEAD"}:
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
        path = await service.download_path(
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
        Route("/api/session", session, methods=["GET", "DELETE"]),
        Route("/api/auth/guest", guest_login, methods=["POST"]),
        Route("/api/auth/github/start", github_start),
        Route("/api/auth/github/callback", github_callback),
        Route("/api/config", config),
        Route("/api/examples/{example_id}/preview", example_preview),
        Route("/api/examples/{example_id}/license", example_license),
        Route("/api/examples/{example_id}", example_open, methods=["POST"]),
        Route("/api/uploads", upload, methods=["POST"]),
        Route("/api/upload-sessions", upload_session_start, methods=["POST"]),
        Route("/api/upload-sessions/{upload_id}", upload_session, methods=["GET", "PUT", "DELETE"]),
        Route(
            "/api/upload-sessions/{upload_id}/complete", upload_session_complete, methods=["POST"]
        ),
        Route("/api/uploads/{upload_id}/file", upload_file),
        Route("/api/uploads/{upload_id}", upload_detail, methods=["GET", "DELETE"]),
        Route("/api/tasks", tasks, methods=["GET", "POST"]),
        Route("/api/tasks/{task_id}", task),
        Route("/api/tasks/{task_id}/cancel", cancel, methods=["POST"]),
        Route("/api/tasks/{task_id}/files/{name}", result_file),
        Mount("/static", StaticFiles(directory=STATIC, check_dir=False)),
    ]
    # Public A2A resources share handlers and validation with the private Web API.
    routes.extend(
        [
            Route("/a2a/config", config),
            Route("/a2a/examples/{example_id}/preview", example_preview),
            Route("/a2a/examples/{example_id}/license", example_license),
            Route("/a2a/examples/{example_id}", example_open, methods=["POST"]),
            Route("/a2a/uploads", upload, methods=["POST"]),
            Route("/a2a/uploads/{upload_id}/file", upload_file),
            Route("/a2a/uploads/{upload_id}", upload_detail, methods=["GET", "DELETE"]),
            Route("/a2a/upload-sessions", upload_session_start, methods=["POST"]),
            Route(
                "/a2a/upload-sessions/{upload_id}", upload_session, methods=["GET", "PUT", "DELETE"]
            ),
            Route(
                "/a2a/upload-sessions/{upload_id}/complete",
                upload_session_complete,
                methods=["POST"],
            ),
            Route("/a2a/tasks/{task_id}/files/{name}", result_file),
        ]
    )
    from medsegagent.a2a import routes as a2a_routes

    routes.extend(a2a_routes(service, optional_identity))
    app = Starlette(
        routes=routes,
        lifespan=lifespan,
        exception_handlers={ServiceError: error, agent.RoutingError: error, AuthError: error},
    )
    app.state.service = service
    app.state.auth = auth

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
