"""Account profile, avatar isolation, and first-use guide regression tests."""
from __future__ import annotations

import atexit
import asyncio
import base64
import io
import logging
import uuid
import sqlite3
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import httpx
from PIL import Image

import src.config as config

_test_runtime = tempfile.TemporaryDirectory(
    prefix=".test-account-profile-", dir=Path(__file__).resolve().parents[2]
)
config.PROJECT_ROOT = Path(_test_runtime.name)

import src.auth as auth
auth.DB_PATH = config.PROJECT_ROOT / "state" / "accounts.sqlite3"

from src.api.main import _normalize_avatar_data_url, app
from src.session.manager import session_manager


@asynccontextmanager
async def _profile_test_lifespan(_app):
    yield


app.router.lifespan_context = _profile_test_lifespan


def _run(coro):
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(coro)


def _close_test_log_handlers() -> None:
    for handler in list(logging.getLogger().handlers):
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
    logging.getLogger().handlers.clear()


atexit.register(_close_test_log_handlers)


def _png_data_url(color: tuple[int, int, int, int] = (244, 142, 38, 255)) -> str:
    image = Image.new("RGBA", (512, 384), color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


@pytest.fixture
def account_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_root = tmp_path / "agent"
    monkeypatch.setattr(config, "PROJECT_ROOT", data_root)
    monkeypatch.setattr(auth, "DB_PATH", data_root / "state" / "accounts.sqlite3")
    auth.init_store()
    return data_root


def test_existing_accounts_migrate_and_keep_identity(account_store: Path):
    # Rebuild an older accounts table to prove profile migration is additive.
    path = account_store / "state" / "accounts.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE users")
        conn.execute("""CREATE TABLE users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL COLLATE NOCASE UNIQUE,
            salt BLOB NOT NULL, password_hash BLOB NOT NULL,
            role TEXT NOT NULL DEFAULT 'user', active INTEGER NOT NULL DEFAULT 1,
            daily_limit INTEGER NOT NULL DEFAULT 30, created_at TEXT NOT NULL
        )""")
        conn.execute("INSERT INTO users(username,salt,password_hash,created_at) VALUES(?,?,?,?)",
                     ("existing_user", b"salt", b"hash", "2026-01-01T00:00:00+00:00"))
    auth.init_store()
    user = auth.user_by_id(1)
    assert user and user["username"] == "existing_user"
    assert user["avatar_preset"] == "01"
    assert user["onboarding_required"] is True


def test_avatar_and_onboarding_state_persist(account_store: Path):
    user, _, error = auth.register("profile_user", "long-enough-password", "20")
    assert error is None and user
    assert user["avatar_preset"] == "20" and user["onboarding_required"] is True

    assert auth.set_custom_avatar(user["user_id"], b"normalized-webp")
    assert auth.get_custom_avatar(user["user_id"]) == b"normalized-webp"
    assert auth.complete_onboarding(user["user_id"])
    assert auth.user_by_id(user["user_id"])["onboarding_required"] is False
    assert auth.set_avatar_preset(user["user_id"], "03")
    assert auth.get_custom_avatar(user["user_id"]) is None
    assert auth.user_by_id(user["user_id"])["avatar_preset"] == "03"


def test_avatar_preset_validation_and_account_deletion(account_store: Path):
    user, _, error = auth.register("avatar_user", "long-enough-password", "21")
    assert user is None and error
    user, _, error = auth.register("avatar_user", "long-enough-password", "01")
    assert user and error is None
    assert not auth.set_avatar_preset(user["user_id"], "../../")
    assert auth.set_custom_avatar(user["user_id"], b"private image")
    assert auth.delete_user_account(user["user_id"])
    assert auth.user_by_id(user["user_id"]) is None
    assert auth.get_custom_avatar(user["user_id"]) is None


def test_avatar_image_is_validated_and_normalized():
    normalized = _normalize_avatar_data_url(_png_data_url())
    with Image.open(io.BytesIO(normalized)) as image:
        assert image.format == "WEBP"
        assert image.size == (256, 256)

    with pytest.raises(Exception) as bad_type:
        _normalize_avatar_data_url("data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=")
    assert getattr(bad_type.value, "status_code", None) == 415

    mismatched_type = "data:image/webp;base64," + _png_data_url().split(",", 1)[1]
    with pytest.raises(Exception) as mismatch:
        _normalize_avatar_data_url(mismatched_type)
    assert getattr(mismatch.value, "status_code", None) == 415

    oversized = "data:image/png;base64," + base64.b64encode(b"x" * (5 * 1024 * 1024 + 1)).decode("ascii")
    with pytest.raises(Exception) as too_large:
        _normalize_avatar_data_url(oversized)
    assert getattr(too_large.value, "status_code", None) == 413


def test_avatar_api_is_account_scoped_and_onboarding_can_be_reopened(
    account_store: Path, monkeypatch: pytest.MonkeyPatch
):
    user_a, token_a, error = auth.register("account_alpha", "long-enough-password", "02")
    user_b, token_b, error_b = auth.register("account_bravo", "long-enough-password", "01")
    assert user_a and token_a and error is None
    assert user_b and token_b and error_b is None
    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            client.cookies.set(auth.COOKIE_NAME, token_a)
            same_origin = {"Origin": "http://testserver"}
            response = await client.get("/api/v2/auth/me")
            assert response.status_code == 200
            assert response.json()["user"]["onboarding_required"] is True
            response = await client.post(
                "/api/v2/auth/avatar/upload", json={"data_url": _png_data_url()}, headers=same_origin)
            assert response.status_code == 200
            assert response.json()["user"]["avatar_custom"] is True
            avatar_response = await client.get("/api/v2/auth/avatar")
            assert avatar_response.status_code == 200
            assert avatar_response.headers["content-type"] == "image/webp"

            response = await client.post("/api/v2/auth/onboarding/complete", headers=same_origin)
            assert response.status_code == 200
            assert response.json()["user"]["onboarding_required"] is False

            client.cookies.set(auth.COOKIE_NAME, token_b)
            assert (await client.get("/api/v2/auth/avatar")).status_code == 404
            assert (await client.get("/api/v2/auth/me")).json()["user"]["avatar_custom"] is False

            client.cookies.set(auth.COOKIE_NAME, token_a)
            import src.research.tools as research_tools
            from src.core.tracing import set_session_id
            monkeypatch.setattr(research_tools, "PROJECT_ROOT", account_store)
            owned_session = str(uuid.uuid4())
            await session_manager.get_or_create_session(
                owned_session, owner_user_id=int(user_a["user_id"]))
            set_session_id(owned_session)
            artifact_url = research_tools._save_artifact(
                "数据审查报告.html", b"<html><body>report</body></html>", "text/html")
            set_session_id("")
            assert (await client.get(f"/api/v2/session/{owned_session}/messages")).status_code == 200
            assert (await client.get(f"/api/v2/session/{owned_session}/citations")).status_code == 200
            assert (await client.get(artifact_url)).status_code == 200

            client.cookies.set(auth.COOKIE_NAME, token_b)
            assert (await client.get(f"/api/v2/session/{owned_session}/messages")).status_code == 404
            assert (await client.get(f"/api/v2/session/{owned_session}/citations")).status_code == 404
            assert (await client.get(artifact_url)).status_code == 404

    _run(scenario())


def test_profile_and_guide_assets_are_present():
    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            for preset in range(1, 21):
                response = await client.get(f"/assets/avatars/{preset:02d}.webp")
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("image/webp")
            html = (await client.get("/")).text
            assert "onboardingTitle" in html
            assert "avatarSettingsOverlay" in html
    _run(scenario())
