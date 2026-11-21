"""Unit and HTTP integration tests for authentication, accounts, sessions, and isolation."""

import re
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

import auth
import server
import workspace
from chat_store import PostgresChatStore
from conftest import EmptyCheckpointer, MemoryObjectStorage


@pytest.fixture
def isolated_env(tmp_path, monkeypatch, postgres_database):
    """Provides isolated PostgreSQL, object storage, workspace, and rate-limit state."""
    original_store = server.chat_store
    original_object_storage = server.object_storage
    original_checkpointer = server.checkpointer
    original_workspace_storage = workspace._workspace_storage
    original_ws_server = server.WORKSPACE_ROOT
    original_ws_mod = workspace.WORKSPACE_ROOT
    original_limiter = server.login_rate_limiter
    original_discover_limiter = server.discover_rate_limiter
    original_otp_cooldown = server.otp_cooldown_limiter
    original_otp_burst = server.otp_burst_limiter

    ws_root = tmp_path / "workspaces"
    ws_root.mkdir(parents=True, exist_ok=True)

    store = PostgresChatStore(postgres_database)
    storage = MemoryObjectStorage()
    monkeypatch.setattr(server, "chat_store", store)
    monkeypatch.setattr(server, "object_storage", storage)
    monkeypatch.setattr(server, "checkpointer", EmptyCheckpointer())
    monkeypatch.setattr(server, "WORKSPACE_ROOT", ws_root)
    monkeypatch.setattr(workspace, "WORKSPACE_ROOT", ws_root)
    workspace.configure_workspace_storage(storage)
    monkeypatch.setattr(server, "login_rate_limiter", auth.LoginRateLimiter(max_attempts=5, window_seconds=900))
    monkeypatch.setattr(server, "discover_rate_limiter", auth.LoginRateLimiter(max_attempts=30, window_seconds=300))
    monkeypatch.setattr(server, "otp_cooldown_limiter", auth.LoginRateLimiter(max_attempts=1, window_seconds=auth.OTP_RESEND_COOLDOWN_SECONDS))
    monkeypatch.setattr(server, "otp_burst_limiter", auth.LoginRateLimiter(max_attempts=10, window_seconds=900))

    try:
        yield store, storage, ws_root
    finally:
        server.chat_store = original_store
        server.object_storage = original_object_storage
        server.checkpointer = original_checkpointer
        workspace.configure_workspace_storage(original_workspace_storage)
        server.WORKSPACE_ROOT = original_ws_server
        workspace.WORKSPACE_ROOT = original_ws_mod
        server.login_rate_limiter = original_limiter
        server.discover_rate_limiter = original_discover_limiter
        server.otp_cooldown_limiter = original_otp_cooldown
        server.otp_burst_limiter = original_otp_burst


def test_password_hashing_supports_long_passwords():
    short_hash = auth.hash_password("CorrectHorse1!")
    assert auth.verify_password("CorrectHorse1!", short_hash)
    assert not auth.verify_password("WrongHorse1!", short_hash)

    long_pw = "A" * 100 + "!9"
    long_hash = auth.hash_password(long_pw)
    assert auth.verify_password(long_pw, long_hash)
    # Must NOT truncate at 72 bytes
    assert not auth.verify_password("A" * 100 + "!0", long_hash)

    with pytest.raises(ValueError):
        auth.hash_password("short")


def test_login_gate_rejects_unauthenticated_requests(isolated_env):
    client = TestClient(server.app)

    # Public routes succeed
    assert client.get("/health").status_code == 200
    sess = client.get("/auth/session")
    assert sess.status_code == 200
    sess_data = sess.json()
    assert sess_data["authenticated"] is False
    assert sess_data["user"] is None
    assert sess_data["has_users"] is False
    assert isinstance(sess_data["email_otp_enabled"], bool)

    # Protected routes reject unauthenticated callers with 401
    for method, path in (
        ("GET", "/threads"),
        ("POST", "/threads"),
        ("GET", "/settings"),
        ("GET", "/providers"),
        ("GET", "/marketplace"),
        ("GET", "/auth/me"),
        ("GET", "/auth/sessions"),
    ):
        res = client.request(method, path)
        assert res.status_code == 401, f"Expected 401 for {method} {path}, got {res.status_code}"

    stream_res = client.post("/chat/stream", json={"message": "hi"})
    assert stream_res.status_code == 401


def test_first_user_becomes_admin_and_claims_legacy_data(isolated_env):
    store, _, _ = isolated_env

    # Pre-seed unowned legacy thread and global settings before any user exists
    legacy_thread = store.create_thread(title="Legacy Conversation")
    store.add_message(legacy_thread["thread_id"], "user", "Old message")
    store.save_settings("gpt-4o", 1)

    client1 = TestClient(server.app)
    res1 = client1.post(
        "/auth/signup",
        json={
            "display_name": "First Admin",
            "email": "Admin@Example.com ",
            "password": "StrongPassword1!",
            "avatar_color": "#20b8cd",
        },
    )
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["is_first_admin"] is True
    assert data1["claimed_legacy_data"] is True
    assert data1["user"]["role"] == "admin"
    assert data1["user"]["email"] == "admin@example.com"
    assert len(data1["recovery_codes"]) == 8
    assert "orbit_session" in client1.cookies

    # Claimed legacy thread and settings appear in first user's account
    threads1 = client1.get("/threads").json()
    assert any(t["thread_id"] == legacy_thread["thread_id"] for t in threads1)
    assert client1.get("/settings").json() == {"model": "gpt-4o", "provider_index": 1}

    # Second registered user becomes 'member' and gets an isolated empty workspace
    client2 = TestClient(server.app)
    res2 = client2.post(
        "/auth/signup",
        json={
            "display_name": "Second Member",
            "email": "member@example.com",
            "password": "StrongPassword2!",
        },
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["is_first_admin"] is False
    assert data2["user"]["role"] == "member"
    assert client2.get("/threads").json() == []
    assert client2.get("/settings").json() == {"model": None, "provider_index": None}


def test_login_logout_and_rate_limiting(isolated_env):
    client = TestClient(server.app)
    client.post(
        "/auth/signup",
        json={
            "display_name": "Rate User",
            "email": "rate@example.com",
            "password": "ValidPassword1!",
        },
    )
    assert client.post("/auth/logout").status_code == 200
    assert client.get("/auth/session").json()["authenticated"] is False

    # 5 failed logins -> 401; 6th attempt -> 429 Too Many Requests
    for _ in range(5):
        r = client.post(
            "/auth/login",
            json={"email": "rate@example.com", "password": "WrongPassword!"},
        )
        assert r.status_code == 401

    locked = client.post(
        "/auth/login",
        json={"email": "rate@example.com", "password": "ValidPassword1!"},
    )
    assert locked.status_code == 429
    assert "Retry-After" in locked.headers

    # Reset limiter and confirm valid login works + cookie persists
    server.login_rate_limiter.reset("login:testclient:rate@example.com")
    ok = client.post(
        "/auth/login",
        json={"email": "rate@example.com", "password": "ValidPassword1!", "remember_me": True},
    )
    assert ok.status_code == 200
    assert client.get("/auth/session").json()["authenticated"] is True


def test_cross_user_isolation(isolated_env):
    client_a = TestClient(server.app)
    client_b = TestClient(server.app)

    client_a.post(
        "/auth/signup",
        json={"display_name": "Alice", "email": "alice@example.com", "password": "PasswordA1!"},
    )
    client_b.post(
        "/auth/signup",
        json={"display_name": "Bob", "email": "bob@example.com", "password": "PasswordB1!"},
    )

    tid = client_a.post("/threads", json={"title": "Alice Secret"}).json()["thread_id"]
    workspace.write_file("secret.py", "print('secret')", thread_id=tid)
    client_a.post(f"/threads/{tid}/canvases", json={"title": "Canvas A", "spec": {"type": "card"}})
    client_a.post(
        f"/threads/{tid}/artifacts",
        json={"title": "Plan A", "artifact_type": "code", "content": "x=1"},
    )

    # Bob cannot read, rename, delete, upload, branch, preview, download, or run Alice's thread
    assert client_b.get(f"/threads/{tid}").status_code == 404
    assert client_b.put(f"/threads/{tid}", json={"title": "Hijacked"}).status_code == 404
    assert client_b.delete(f"/threads/{tid}").status_code == 404
    assert client_b.get(f"/threads/{tid}/canvases").status_code == 404
    assert client_b.get(f"/threads/{tid}/artifacts").status_code == 404
    assert client_b.get(f"/threads/{tid}/workspace/files").status_code == 404
    assert client_b.get(f"/threads/{tid}/workspace/preview?path=secret.py").status_code == 404
    assert client_b.get(f"/threads/{tid}/workspace/raw?path=secret.py").status_code == 404
    assert client_b.get(f"/threads/{tid}/workspace/download?path=secret.py").status_code == 404
    assert (
        client_b.post(f"/threads/{tid}/workspace/run", json={"command_or_file": "secret.py"}).status_code
        == 404
    )

    # Alice still has full access
    assert client_a.get(f"/threads/{tid}").status_code == 200
    assert client_a.get(f"/threads/{tid}/workspace/raw?path=secret.py").status_code == 200

    # Per-user marketplace skills and SPA route checks
    server.marketplace.register_custom_skill({
        "id": "isolated-skill",
        "name": "Isolated Skill",
        "category": "Workflow & Standards",
        "description": "User scoped skill",
        "type": "skill",
    })
    try:
        assert (
            client_a.post(
                "/marketplace/skills/isolated-skill/toggle", json={"install": True}
            ).status_code
            == 200
        )
        alice_skills = client_a.get("/marketplace").json()["skills"]
        bob_skills = client_b.get("/marketplace").json()["skills"]
        assert alice_skills[0]["is_installed"] is True
        assert bob_skills[0]["is_installed"] is False
    finally:
        server.marketplace.clear_all_skills()

    # SPA routes render HTML while unknown /auth API paths return JSON 404
    assert client_a.get("/login").status_code == 200
    assert "text/html" in client_a.get("/signup").headers["content-type"]
    assert client_a.get("/auth/nonexistent").status_code == 404


def test_profile_password_sessions_and_recovery_codes(isolated_env):
    client1 = TestClient(server.app)
    client2 = TestClient(server.app)

    signup = client1.post(
        "/auth/signup",
        json={"display_name": "Sam", "email": "sam@example.com", "password": "OldPassword1!"},
    ).json()
    codes = signup["recovery_codes"]

    # Second concurrent session for the same user
    client2.post(
        "/auth/login",
        json={"email": "sam@example.com", "password": "OldPassword1!"},
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0) Firefox/128.0"},
    )
    sessions = client1.get("/auth/sessions").json()["sessions"]
    assert len(sessions) == 2

    # Email change requires current password
    bad_email_update = client1.patch(
        "/auth/profile",
        json={"display_name": "Sam Updated", "email": "sam2@example.com", "avatar_color": "#0ea5e9"},
    )
    assert bad_email_update.status_code == 403

    good_profile = client1.patch(
        "/auth/profile",
        json={
            "display_name": "Sam Updated",
            "email": "sam2@example.com",
            "avatar_color": "#0ea5e9",
            "current_password": "OldPassword1!",
        },
    )
    assert good_profile.status_code == 200
    assert good_profile.json()["user"]["display_name"] == "Sam Updated"
    assert good_profile.json()["user"]["email"] == "sam2@example.com"

    # Password change revokes client2 session when requested
    pw_res = client1.post(
        "/auth/password",
        json={
            "current_password": "OldPassword1!",
            "new_password": "NewPassword2!",
            "revoke_other_sessions": True,
        },
    )
    assert pw_res.status_code == 200
    assert pw_res.json()["revoked_other_sessions"] == 1
    assert client2.get("/auth/me").status_code == 401
    assert client1.get("/auth/me").status_code == 200

    # Local password recovery using single-use code (also invalidates remaining session)
    reset_res = client2.post(
        "/auth/recovery/reset",
        json={
            "email": "sam2@example.com",
            "recovery_code": codes[0].lower().replace("-", " "),
            "new_password": "RecoveredPassword3!",
        },
    )
    assert reset_res.status_code == 200
    assert client1.get("/auth/me").status_code == 401

    # Reusing the same recovery code must fail
    reuse_res = client2.post(
        "/auth/recovery/reset",
        json={
            "email": "sam2@example.com",
            "recovery_code": codes[0],
            "new_password": "AnotherPassword4!",
        },
    )
    assert reuse_res.status_code == 400

    # Login with recovered password succeeds and allows code regeneration
    assert (
        client2.post(
            "/auth/login",
            json={"email": "sam2@example.com", "password": "RecoveredPassword3!"},
        ).status_code
        == 200
    )
    regen = client2.post(
        "/auth/recovery-codes/regenerate", json={"password": "RecoveredPassword3!"}
    )
    assert regen.status_code == 200
    assert len(regen.json()["recovery_codes"]) == 8


def test_admin_user_controls_and_account_deletion_cascade(isolated_env):
    _, storage, ws_root = isolated_env
    admin_client = TestClient(server.app)
    member_client = TestClient(server.app)

    admin_res = admin_client.post(
        "/auth/signup",
        json={"display_name": "Admin", "email": "admin@corp.com", "password": "Password1!"},
    ).json()
    member_res = member_client.post(
        "/auth/signup",
        json={"display_name": "Member", "email": "member@corp.com", "password": "Password2!"},
    ).json()
    admin_id = admin_res["user"]["user_id"]
    member_id = member_res["user"]["user_id"]

    # Non-admin blocked from admin endpoints
    assert member_client.get("/auth/admin/users").status_code == 403

    # Admin lists users and cannot demote self
    users = admin_client.get("/auth/admin/users").json()["users"]
    assert len(users) == 2
    assert (
        admin_client.patch(f"/auth/admin/users/{admin_id}", json={"role": "member"}).status_code
        == 400
    )

    # Admin issues temporary reset code for member
    issued = admin_client.post(f"/auth/admin/users/{member_id}/reset-code")
    assert issued.status_code == 200
    temp_code = issued.json()["recovery_code"]

    # Admin disables member -> revokes member sessions immediately
    assert (
        admin_client.patch(f"/auth/admin/users/{member_id}", json={"is_active": False}).status_code
        == 200
    )
    assert member_client.get("/auth/me").status_code == 401
    assert (
        member_client.post(
            "/auth/login", json={"email": "member@corp.com", "password": "Password2!"}
        ).status_code
        == 403
    )

    # Re-enable member and test reset using admin-issued code
    assert (
        admin_client.patch(f"/auth/admin/users/{member_id}", json={"is_active": True}).status_code
        == 200
    )
    assert (
        member_client.post(
            "/auth/recovery/reset",
            json={
                "email": "member@corp.com",
                "recovery_code": temp_code,
                "new_password": "ResetByAdminCode1!",
            },
        ).status_code
        == 200
    )
    assert (
        member_client.post(
            "/auth/login",
            json={"email": "member@corp.com", "password": "ResetByAdminCode1!"},
        ).status_code
        == 200
    )

    # Member creates thread, workspace file, and upload, then deletes account
    tid = member_client.post("/threads", json={"title": "Member Thread"}).json()["thread_id"]
    workspace.write_file("temp.py", "print('bye')", thread_id=tid)
    up = member_client.post(
        f"/threads/{tid}/attachments?filename=note.txt",
        content=b"delete me",
        headers={"Content-Type": "text/plain"},
    )
    assert up.status_code == 200
    ws_dir = ws_root / tid
    assert ws_dir.is_dir()
    assert len(storage.objects) == 1

    # Wrong password rejected
    assert (
        member_client.post("/auth/account/delete", json={"password": "WrongPassword!"}).status_code
        == 403
    )
    # Sole admin deletion blocked while another user exists
    assert (
        admin_client.post("/auth/account/delete", json={"password": "Password1!"}).status_code
        == 409
    )

    # Member deletes own account -> DB rows, upload file, and workspace directory all removed
    del_ok = member_client.post(
        "/auth/account/delete", json={"password": "ResetByAdminCode1!"}
    )
    assert del_ok.status_code == 200
    assert not ws_dir.exists()
    assert storage.objects == {}
    assert member_client.get("/auth/session").json()["authenticated"] is False


def test_email_discover_and_rate_limit(isolated_env, monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    client = TestClient(server.app)

    sess = client.get("/auth/session").json()
    assert sess["email_otp_enabled"] is False
    assert sess["has_users"] is False

    # Before first signup -> exists=False, is_first_account=True
    d1 = client.post("/auth/discover", json={"email": "  FirstAdmin@Example.com  "})
    assert d1.status_code == 200
    data1 = d1.json()
    assert data1["email"] == "firstadmin@example.com"
    assert data1["exists"] is False
    assert data1["is_first_account"] is True
    assert "display_name" not in data1

    # Invalid email -> 422
    assert client.post("/auth/discover", json={"email": "not-an-email"}).status_code == 422

    # Sign up user
    signup = client.post(
        "/auth/signup",
        json={
            "display_name": "First Admin",
            "email": "firstadmin@example.com",
            "password": "StrongPassword1!",
        },
    )
    assert signup.status_code == 200

    # Existing email now reports exists=True, unknown email exists=False & is_first_account=False
    d2 = client.post("/auth/discover", json={"email": "FIRSTADMIN@example.com"}).json()
    assert d2["exists"] is True
    assert d2["has_users"] is True
    assert d2["is_first_account"] is False

    d3 = client.post("/auth/discover", json={"email": "second@example.com"}).json()
    assert d3["exists"] is False
    assert d3["is_first_account"] is False

    # Tight rate limit check (4 already used above -> burn remaining 26 out of 30)
    for _ in range(26):
        assert client.post("/auth/discover", json={"email": "probe@example.com"}).status_code == 200
    blocked = client.post("/auth/discover", json={"email": "probe@example.com"})
    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers


def test_email_otp_disabled_when_smtp_unconfigured(isolated_env, monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    client = TestClient(server.app)

    assert (
        client.post("/auth/otp/request", json={"email": "a@example.com", "purpose": "login"}).status_code
        == 400
    )
    assert (
        client.post("/auth/otp/verify-login", json={"email": "a@example.com", "code": "123456"}).status_code
        == 400
    )
    assert (
        client.post(
            "/auth/otp/reset-password",
            json={"email": "a@example.com", "code": "123456", "new_password": "NewPassword1!"},
        ).status_code
        == 400
    )


def test_email_otp_login_and_password_reset_flow(isolated_env, monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.local")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_FROM", "noreply@example.local")

    sent_messages = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=15):
            self.host = host
            self.port = port

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def ehlo(self):
            return (250, b"ok")

        def starttls(self):
            return (220, b"ok")

        def login(self, user, password):
            return (235, b"ok")

        def send_message(self, msg):
            sent_messages.append(
                {
                    "to": msg["To"],
                    "subject": msg["Subject"],
                    "body": msg.get_content(),
                }
            )

    monkeypatch.setattr(auth.smtplib, "SMTP", FakeSMTP)

    client1 = TestClient(server.app)
    client2 = TestClient(server.app)

    assert client1.get("/auth/session").json()["email_otp_enabled"] is True

    signup = client1.post(
        "/auth/signup",
        json={
            "display_name": "Alice OTP",
            "email": "alice@example.com",
            "password": "InitialPassword1!",
        },
    )
    assert signup.status_code == 200

    # Request login OTP for Alice
    req_res = client2.post(
        "/auth/otp/request", json={"email": "alice@example.com", "purpose": "login"}
    )
    assert req_res.status_code == 200
    assert len(sent_messages) == 1
    match = re.search(r"\b(\d{6})\b", sent_messages[-1]["body"])
    assert match is not None
    code1 = match.group(1)

    # Immediate repeat hits 30s cooldown
    cooldown_res = client2.post(
        "/auth/otp/request", json={"email": "alice@example.com", "purpose": "login"}
    )
    assert cooldown_res.status_code == 429

    # 5 wrong attempts invalidate code1 (even though login_rate_limiter also tracks attempts)
    wrong_code = "000000" if code1 != "000000" else "999999"
    for _ in range(5):
        bad_verify = client2.post(
            "/auth/otp/verify-login",
            json={"email": "alice@example.com", "code": wrong_code},
        )
        assert bad_verify.status_code == 401

    # Fresh limiters to simulate waiting out cooldown and verify code1 was invalidated after 5 bad guesses
    monkeypatch.setattr(server, "login_rate_limiter", auth.LoginRateLimiter(max_attempts=5, window_seconds=900))
    monkeypatch.setattr(server, "otp_cooldown_limiter", auth.LoginRateLimiter(max_attempts=1, window_seconds=30))

    assert (
        client2.post(
            "/auth/otp/verify-login", json={"email": "alice@example.com", "code": code1}
        ).status_code
        == 401
    )

    # Request a fresh login code and sign in cleanly
    assert (
        client2.post(
            "/auth/otp/request", json={"email": "alice@example.com", "purpose": "login"}
        ).status_code
        == 200
    )
    code2 = re.search(r"\b(\d{6})\b", sent_messages[-1]["body"]).group(1)
    ok_verify = client2.post(
        "/auth/otp/verify-login",
        json={"email": "alice@example.com", "code": code2, "remember_me": True},
    )
    assert ok_verify.status_code == 200
    assert ok_verify.json()["user"]["email"] == "alice@example.com"
    assert client2.get("/auth/me").status_code == 200

    # Reusing the consumed code fails
    assert (
        client2.post(
            "/auth/otp/verify-login", json={"email": "alice@example.com", "code": code2}
        ).status_code
        == 401
    )

    # Recovery via email OTP resets password and revokes both client1 & client2 sessions
    assert (
        client2.post(
            "/auth/otp/request", json={"email": "alice@example.com", "purpose": "recovery"}
        ).status_code
        == 200
    )
    recovery_code = re.search(r"\b(\d{6})\b", sent_messages[-1]["body"]).group(1)

    reset_res = client2.post(
        "/auth/otp/reset-password",
        json={
            "email": "alice@example.com",
            "code": recovery_code,
            "new_password": "BrandNewPassword9!",
        },
    )
    assert reset_res.status_code == 200
    assert client1.get("/auth/me").status_code == 401
    assert client2.get("/auth/me").status_code == 401

    # New password works and logout-all(keep_current=False) signs out cleanly
    login_new = client2.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "BrandNewPassword9!"},
    )
    assert login_new.status_code == 200
    logout_all = client2.post("/auth/logout-all", json={"keep_current": False})
    assert logout_all.status_code == 200
    assert logout_all.json()["signed_out_current"] is True
    assert client2.get("/auth/me").status_code == 401
