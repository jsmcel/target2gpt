from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import time
from typing import Any


ROOT = Path(__file__).resolve().parent
SECRETS_DIR = ROOT / "secrets"
OUTPUT_DIR = ROOT / "output"
USERS_PATH = SECRETS_DIR / "access_users.json"
ACCESS_REQUESTS_PATH = OUTPUT_DIR / "access_requests.jsonl"

PBKDF2_ITERATIONS = 260_000


def normalize_email(value: str | None) -> str:
    return str(value or "").strip().lower()


def valid_email(value: str) -> bool:
    email = normalize_email(value)
    return "@" in email and "." in email.rsplit("@", 1)[-1] and " " not in email


def _empty_store() -> dict[str, Any]:
    return {"version": 1, "users": {}}


def load_user_store() -> dict[str, Any]:
    if not USERS_PATH.exists():
        return _empty_store()
    try:
        data = json.loads(USERS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _empty_store()
    if not isinstance(data, dict):
        return _empty_store()
    users = data.get("users")
    if not isinstance(users, dict):
        data["users"] = {}
    return data


def save_user_store(data: dict[str, Any]) -> None:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = USERS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, USERS_PATH)


def get_user(email: str) -> dict[str, Any] | None:
    store = load_user_store()
    user = store.get("users", {}).get(normalize_email(email))
    return user if isinstance(user, dict) else None


def approved_user(email: str) -> dict[str, Any] | None:
    user = get_user(email)
    if not user or user.get("status") != "approved":
        return None
    return user


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    if not password:
        raise ValueError("Password cannot be empty")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str | None) -> bool:
    if not password or not encoded:
        return False
    try:
        scheme, iterations_raw, salt_raw, digest_raw = encoded.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        salt = _b64_decode(salt_raw)
        expected = _b64_decode(digest_raw)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    except Exception:
        return False
    return hmac.compare_digest(actual, expected)


def generate_temporary_key() -> str:
    return secrets.token_urlsafe(18)


def approve_user(email: str, *, name: str = "", temporary_key: str | None = None) -> tuple[dict[str, Any], str]:
    normalized = normalize_email(email)
    if not valid_email(normalized):
        raise ValueError("Invalid email")
    key = temporary_key or generate_temporary_key()
    now = int(time.time())
    store = load_user_store()
    users = store.setdefault("users", {})
    previous = users.get(normalized, {}) if isinstance(users.get(normalized), dict) else {}
    user = {
        **previous,
        "email": normalized,
        "name": name or previous.get("name") or normalized,
        "status": "approved",
        "password_hash": hash_password(key),
        "must_change_password": True,
        "approved_at": now,
        "updated_at": now,
    }
    user.setdefault("created_at", now)
    users[normalized] = user
    save_user_store(store)
    return user, key


def set_user_password(email: str, password: str, *, must_change_password: bool) -> dict[str, Any]:
    normalized = normalize_email(email)
    store = load_user_store()
    users = store.setdefault("users", {})
    user = users.get(normalized)
    if not isinstance(user, dict):
        raise KeyError(normalized)
    user["password_hash"] = hash_password(password)
    user["must_change_password"] = bool(must_change_password)
    user["updated_at"] = int(time.time())
    users[normalized] = user
    save_user_store(store)
    return user


def revoke_user(email: str) -> dict[str, Any]:
    normalized = normalize_email(email)
    store = load_user_store()
    users = store.setdefault("users", {})
    user = users.get(normalized)
    if not isinstance(user, dict):
        raise KeyError(normalized)
    user["status"] = "revoked"
    user["updated_at"] = int(time.time())
    users[normalized] = user
    save_user_store(store)
    return user


def list_users() -> list[dict[str, Any]]:
    users = load_user_store().get("users", {})
    return sorted(
        (user for user in users.values() if isinstance(user, dict)),
        key=lambda item: str(item.get("email") or ""),
    )


def load_access_requests() -> list[dict[str, Any]]:
    if not ACCESS_REQUESTS_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in ACCESS_REQUESTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def list_latest_access_requests() -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for item in load_access_requests():
        email = normalize_email(item.get("email") or item.get("google_email"))
        if not email:
            continue
        previous = latest.get(email)
        if not previous or int(item.get("ts") or 0) >= int(previous.get("ts") or 0):
            latest[email] = item
    users = load_user_store().get("users", {})
    result = []
    for email, item in latest.items():
        user = users.get(email, {}) if isinstance(users, dict) else {}
        result.append(
            {
                **item,
                "email": email,
                "approval_status": user.get("status") if isinstance(user, dict) else None,
                "must_change_password": bool(user.get("must_change_password")) if isinstance(user, dict) else False,
            }
        )
    return sorted(result, key=lambda item: int(item.get("ts") or 0), reverse=True)


def record_access_request(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ACCESS_REQUESTS_PATH.open("a", encoding="utf-8").write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    )


