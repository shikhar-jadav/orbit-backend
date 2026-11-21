"""Authentication, session management, recovery codes, and rate limiting for Orbit."""

import base64
import hashlib
import os
import re
import secrets
import smtplib
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any, Callable, Deque, Dict, Optional

import bcrypt
from fastapi import Depends, HTTPException, Request, Response

COOKIE_NAME = "orbit_session"
SESSION_TTL_SECONDS = 24 * 60 * 60          # 24 hours
REMEMBER_ME_TTL_SECONDS = 30 * 24 * 60 * 60 # 30 days
OTP_TTL_SECONDS = 10 * 60                   # 10 minutes
OTP_RESEND_COOLDOWN_SECONDS = 30

AVATAR_PALETTE = (
    "#673ab7",
    "#20b8cd",
    "#0ea5e9",
    "#10b981",
    "#f59e0b",
    "#ec4899",
    "#6366f1",
    "#ef4444",
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_RECOVERY_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def validate_email(email: str) -> str:
    normalized = normalize_email(email)
    if not normalized or len(normalized) > 254 or not _EMAIL_RE.match(normalized):
        raise ValueError("Enter a valid email address.")
    return normalized


def validate_display_name(display_name: str) -> str:
    cleaned = " ".join((display_name or "").split()).strip()
    if not cleaned:
        raise ValueError("Display name cannot be blank.")
    if len(cleaned) > 64:
        raise ValueError("Display name must be 64 characters or fewer.")
    return cleaned


def validate_avatar_color(color: Optional[str]) -> str:
    candidate = (color or "").strip()
    if _HEX_COLOR_RE.match(candidate):
        return candidate.lower()
    return AVATAR_PALETTE[0]


def validate_password(password: str) -> str:
    if not isinstance(password, str) or len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if len(password) > 128:
        raise ValueError("Password must be 128 characters or fewer.")
    return password


def _prehash_password(password: str) -> bytes:
    """Pre-hashes arbitrary-length UTF-8 passwords cleanly before bcrypt."""
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    validate_password(password)
    hashed = bcrypt.hashpw(_prehash_password(password), bcrypt.gensalt(rounds=12))
    return hashed.decode("ascii")


def verify_password(password: str, stored_hash: str) -> bool:
    if not password or not stored_hash:
        return False
    try:
        return bcrypt.checkpw(_prehash_password(password), stored_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False


def generate_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def compute_session_expiry(remember_me: bool) -> tuple[str, int]:
    ttl = REMEMBER_ME_TTL_SECONDS if remember_me else SESSION_TTL_SECONDS
    expires_dt = datetime.now(timezone.utc) + timedelta(seconds=ttl)
    return expires_dt.isoformat(timespec="seconds"), ttl


def normalize_recovery_code(code: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", (code or "")).upper()


def hash_recovery_code(code: str) -> str:
    normalized = normalize_recovery_code(code)
    return hashlib.sha256(f"orbit-recovery:{normalized}".encode("utf-8")).hexdigest()


def generate_recovery_codes(count: int = 8) -> list[str]:
    codes: list[str] = []
    for _ in range(count):
        part1 = "".join(secrets.choice(_RECOVERY_ALPHABET) for _ in range(4))
        part2 = "".join(secrets.choice(_RECOVERY_ALPHABET) for _ in range(4))
        codes.append(f"ORBIT-{part1}-{part2}")
    return codes


def generate_otp_code(digits: int = 6) -> str:
    return "".join(str(secrets.randbelow(10)) for _ in range(digits))


def normalize_otp_code(code: str) -> str:
    return re.sub(r"\D", "", code or "")


def hash_otp_code(email: str, purpose: str, code: str) -> str:
    normalized_email = normalize_email(email)
    normalized_code = normalize_otp_code(code)
    return hashlib.sha256(
        f"orbit-otp:{purpose}:{normalized_email}:{normalized_code}".encode("utf-8")
    ).hexdigest()


def compute_otp_expiry() -> str:
    expires_dt = datetime.now(timezone.utc) + timedelta(seconds=OTP_TTL_SECONDS)
    return expires_dt.isoformat(timespec="seconds")


def is_email_otp_enabled() -> bool:
    return bool(os.getenv("SMTP_HOST", "").strip())


def send_otp_email(to_email: str, code: str, purpose: str) -> None:
    """Sends a 6-digit verification code via SMTP configured in .env."""
    host = os.getenv("SMTP_HOST", "").strip()
    if not host:
        raise RuntimeError("SMTP_HOST is not configured.")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    sender = os.getenv("SMTP_FROM", "").strip() or user or "noreply@orbit.local"

    action_label = "sign in to" if purpose == "login" else "reset your password on"
    msg = EmailMessage()
    msg["Subject"] = f"{code} is your Orbit Agent Studio verification code"
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(
        f"Your temporary Orbit Agent Studio verification code is: {code}\n\n"
        f"Enter this code to {action_label} your Orbit workspace.\n"
        f"This code expires in {OTP_TTL_SECONDS // 60} minutes.\n\n"
        "If you did not request this code, you can safely ignore this email.\n"
    )

    use_ssl = os.getenv("SMTP_SSL", "").lower() in {"1", "true", "yes"} or port == 465
    if use_ssl:
        with smtplib.SMTP_SSL(host, port, timeout=15) as smtp:
            if user and password:
                smtp.login(user, password)
            smtp.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.ehlo()
            if os.getenv("SMTP_TLS", "true").lower() not in {"0", "false", "no"}:
                try:
                    smtp.starttls()
                    smtp.ehlo()
                except smtplib.SMTPNotSupportedError:
                    pass
            if user and password:
                smtp.login(user, password)
            smtp.send_message(msg)


class LoginRateLimiter:
    """Thread-safe sliding-window rate limiter with Upstash Redis distributed sync."""

    def __init__(self, max_attempts: int = 5, window_seconds: int = 900):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._failures: dict[str, Deque[float]] = {}
        self._lock = threading.Lock()
        try:
            from redis_cache import DistributedRateLimiter
            self._dist_limiter = DistributedRateLimiter(max_attempts=max_attempts, window_seconds=window_seconds)
        except Exception:
            self._dist_limiter = None

    def _prune(self, dq: Deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while dq and dq[0] <= cutoff:
            dq.popleft()

    def check(self, key: str) -> tuple[bool, int]:
        if self._dist_limiter:
            try:
                allowed, retry_after = self._dist_limiter.check(key)
                if not allowed:
                    return allowed, retry_after
            except Exception:
                pass
        now = time.monotonic()
        with self._lock:
            dq = self._failures.get(key)
            if not dq:
                return True, 0
            self._prune(dq, now)
            if len(dq) >= self.max_attempts:
                retry_after = max(1, int(self.window_seconds - (now - dq[0])) + 1)
                return False, retry_after
            return True, 0

    def record_failure(self, key: str) -> None:
        if self._dist_limiter:
            try:
                self._dist_limiter.record_failure(key)
            except Exception:
                pass
        now = time.monotonic()
        with self._lock:
            dq = self._failures.setdefault(key, deque())
            self._prune(dq, now)
            dq.append(now)

    def reset(self, key: str) -> None:
        if self._dist_limiter:
            try:
                self._dist_limiter.reset(key)
            except Exception:
                pass
        with self._lock:
            self._failures.pop(key, None)


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "local"


def parse_user_agent(user_agent: str) -> str:
    ua = (user_agent or "").lower()
    if not ua:
        return "Web Browser"
    browser = "Browser"
    if "edg/" in ua:
        browser = "Edge"
    elif "chrome/" in ua and "chromium" not in ua:
        browser = "Chrome"
    elif "firefox/" in ua:
        browser = "Firefox"
    elif "safari/" in ua and "chrome/" not in ua:
        browser = "Safari"
    elif "curl/" in ua or "python" in ua or "httpx" in ua:
        return "API Client"

    os_name = ""
    if "windows" in ua:
        os_name = "Windows"
    elif "mac os" in ua or "macintosh" in ua:
        os_name = "macOS"
    elif "iphone" in ua or "ipad" in ua:
        os_name = "iOS"
    elif "android" in ua:
        os_name = "Android"
    elif "linux" in ua:
        os_name = "Linux"

    return f"{browser} on {os_name}" if os_name else browser


def _is_secure_cookie(request: Request) -> bool:
    if os.getenv("COOKIE_SECURE", "").lower() in {"1", "true", "yes"}:
        return True
    proto = request.headers.get("x-forwarded-proto", "").lower()
    return request.url.scheme == "https" or proto == "https"


def _cookie_samesite(request: Request) -> str:
    env_samesite = os.getenv("COOKIE_SAMESITE", "").lower().strip()
    if env_samesite in {"none", "lax", "strict"}:
        return env_samesite
    if _is_secure_cookie(request):
        return "none"
    return "lax"


def set_session_cookie(
    response: Response, token: str, max_age_seconds: int, request: Request
) -> None:
    samesite = _cookie_samesite(request)
    secure = _is_secure_cookie(request) or samesite == "none"
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=max_age_seconds,
        httponly=True,
        samesite=samesite,
        secure=secure,
        path="/",
    )


def clear_session_cookie(response: Response, request: Request) -> None:
    samesite = _cookie_samesite(request)
    secure = _is_secure_cookie(request) or samesite == "none"
    response.delete_cookie(
        key=COOKIE_NAME,
        httponly=True,
        samesite=samesite,
        secure=secure,
        path="/",
    )


def extract_session_token(request: Request) -> Optional[str]:
    cookie_token = request.cookies.get(COOKIE_NAME)
    if cookie_token:
        return cookie_token.strip()
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        bearer = auth_header[7:].strip()
        if bearer:
            return bearer
    return None


def make_auth_dependencies(get_store: Callable[[], Any]):
    """Builds FastAPI auth dependencies referencing the active PostgreSQL chat store."""

    async def get_current_user_optional(request: Request) -> Optional[Dict[str, Any]]:
        token = extract_session_token(request)
        if not token:
            return None
        token_h = hash_token(token)
        try:
            from redis_cache import get_cached_session, set_cached_session
            cached = get_cached_session(token_h)
            if cached:
                return cached
        except Exception:
            pass

        store = get_store()
        session_info = store.get_session_with_user(token_h)
        if not session_info:
            return None
        user = dict(session_info["user"])
        user["session_id"] = session_info["session_id"]
        user["remember_me"] = session_info["remember_me"]
        try:
            set_cached_session(token_h, user)
        except Exception:
            pass
        return user

    async def require_user(
        current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    ) -> Dict[str, Any]:
        if current_user is None:
            raise HTTPException(
                status_code=401,
                detail="Authentication required. Please sign in.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return current_user

    async def require_admin(
        current_user: Dict[str, Any] = Depends(require_user),
    ) -> Dict[str, Any]:
        if current_user.get("role") != "admin":
            raise HTTPException(
                status_code=403,
                detail="Administrator privileges required.",
            )
        return current_user

    return get_current_user_optional, require_user, require_admin


def _cli_main() -> None:
    import argparse
    import getpass
    from chat_store import PostgresChatStore
    from database import PostgresDatabase
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Orbit Auth local administration CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    reset_cmd = sub.add_parser("reset-password", help="Reset a local user's password")
    reset_cmd.add_argument("email", help="Account email address")
    args = parser.parse_args()

    load_dotenv()
    database = PostgresDatabase()
    database.open()
    try:
        database.migrate()
        store = PostgresChatStore(database)

        if args.command == "reset-password":
            user = store.get_user_credentials_by_email(normalize_email(args.email))
            if not user:
                print(f"No user found with email: {args.email}")
                raise SystemExit(1)
            new_pw = getpass.getpass("New password (min 8 chars): ")
            confirm_pw = getpass.getpass("Confirm password: ")
            if new_pw != confirm_pw:
                print("Passwords do not match.")
                raise SystemExit(1)
            pw_hash = hash_password(new_pw)
            store.update_user_password(user["user_id"], pw_hash)
            store.revoke_all_sessions(user["user_id"])
            print(f"Password updated and all sessions revoked for {user['email']}.")
    finally:
        database.close()


if __name__ == "__main__":
    _cli_main()
