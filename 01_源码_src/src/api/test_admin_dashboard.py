import atexit
import asyncio
import logging
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

import src.config as config

_test_data = tempfile.TemporaryDirectory(
    prefix=".test-admin-route-", dir=Path(__file__).resolve().parents[2]
)
config.PROJECT_ROOT = Path(_test_data.name)

import src.auth as auth
from src.api.main import app


def _close_test_log_handlers() -> None:
    """Close the app's file logger before TemporaryDirectory removes its sink."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
    root.handlers.clear()


atexit.register(_close_test_log_handlers)


@asynccontextmanager
async def _test_lifespan(_app):
    # Route authorization tests should not initialize RAG models or background
    # cleanup tasks; those belong to deployment smoke tests.
    yield


app.router.lifespan_context = _test_lifespan


def _run(coro):
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(coro)


def test_admin_page_redirects_anonymous_users(monkeypatch):
    monkeypatch.setattr(auth, "user_for_token", lambda _token: None)
    response = _request("GET", "/admin", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/?next=%2Fadmin"


def test_admin_page_denies_regular_accounts(monkeypatch):
    regular_user = {"user_id": 7, "username": "member", "role": "user", "active": True}
    monkeypatch.setattr(auth, "user_for_token", lambda token: regular_user if token == "member-session" else None)
    response = _request("GET", "/admin", cookies={"citrus_auth": "member-session"})

    assert response.status_code == 403
    assert "仅管理员可访问" in response.text


def test_admin_page_serves_dashboard_to_active_admin(monkeypatch):
    admin_user = {"user_id": 1, "username": "owner", "role": "admin", "active": True}
    monkeypatch.setattr(auth, "user_for_token", lambda token: admin_user if token == "admin-session" else None)
    response = _request("GET", "/admin", cookies={"citrus_auth": "admin-session"})

    assert response.status_code == 200
    assert "CitrusQA 管理后台" in response.text
    assert response.headers["cache-control"] == "no-store"


def test_admin_api_remains_server_protected(monkeypatch):
    regular_user = {"user_id": 7, "username": "member", "role": "user", "active": True}
    monkeypatch.setattr(auth, "user_for_token", lambda token: regular_user if token == "member-session" else None)
    response = _request(
        "GET", "/api/v2/admin/queue", cookies={"citrus_auth": "member-session"})

    assert response.status_code == 403


def test_client_ip_uses_valid_cloudflare_header_only_when_enabled(monkeypatch):
    from starlette.requests import Request
    from src.api.main import _client_ip

    scope = {
        "type": "http", "method": "GET", "path": "/", "headers": [(b"cf-connecting-ip", b"203.0.113.8")],
        "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 8000), "scheme": "http",
        "query_string": b"", "http_version": "1.1",
    }
    request = Request(scope)
    monkeypatch.setenv("CITRUS_TRUST_CLOUDFLARE_PROXY", "1")
    assert _client_ip(request) == "203.0.113.8"
    monkeypatch.setenv("CITRUS_TRUST_CLOUDFLARE_PROXY", "0")
    assert _client_ip(request) == "127.0.0.1"


async def _async_request(method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    cookies = kwargs.pop("cookies", None)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver",
                                 follow_redirects=False) as client:
        if cookies:
            client.cookies.update(cookies)
        return await client.request(method, path, **kwargs)


def _request(method: str, path: str, **kwargs):
    return _run(_async_request(method, path, **kwargs))
