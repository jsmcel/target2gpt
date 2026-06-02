#!/usr/bin/env python
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from email.message import EmailMessage
import hashlib
import hmac
import html
import json
import os
from pathlib import Path
import secrets
import smtplib
import ssl
import threading
import time
from typing import Any
from urllib.parse import urlencode
import uuid

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from target2_access import (
    ACCESS_REQUESTS_PATH,
    approved_user,
    get_user,
    normalize_email,
    record_access_request,
    list_latest_access_requests,
    set_user_password,
    valid_email,
    verify_password,
)
from target2_ask import INDEX_PATH, MAX_CONTEXT_HITS, answer_question, augment_hits, build_codex_context, expand_neighbor_hits, load_index, load_json, rerank_context_hits, retrieve


ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_google_oauth_client_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    client_type = "web" if data.get("web") else "installed" if data.get("installed") else "raw"
    client = data.get("web") or data.get("installed") or data
    client_id = str(client.get("client_id") or "").strip()
    client_secret = str(client.get("client_secret") or "").strip()
    if client_id and "GOOGLE_CLIENT_ID" not in os.environ:
        os.environ["GOOGLE_CLIENT_ID"] = client_id
    if client_secret and "GOOGLE_CLIENT_SECRET" not in os.environ:
        os.environ["GOOGLE_CLIENT_SECRET"] = client_secret
    if "GOOGLE_CLIENT_TYPE" not in os.environ:
        os.environ["GOOGLE_CLIENT_TYPE"] = client_type


load_env_file(ROOT / "secrets" / "target2_oauth.env")
load_google_oauth_client_file(Path(os.environ.get("GOOGLE_CLIENT_SECRETS_FILE", ROOT / "secrets" / "google_oauth_client.json")))

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_CLIENT_TYPE = os.environ.get("GOOGLE_CLIENT_TYPE", "").strip()
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "").strip()
TARGET2_PUBLIC_BASE_URL = os.environ.get("TARGET2_PUBLIC_BASE_URL", "").strip().rstrip("/")
SESSION_COOKIE = "target2_session"
OAUTH_STATE_COOKIE = "target2_oauth_state"
SESSION_MAX_AGE = 12 * 60 * 60
OAUTH_STATE_MAX_AGE = 10 * 60
SESSION_SECRET_PATH = ROOT / "secrets" / "target2_session_secret.txt"
COOKIE_SECURE = os.environ.get("TARGET2_COOKIE_SECURE", "1").lower() not in {"0", "false", "no", "off"}
COOKIE_SAMESITE = os.environ.get("TARGET2_COOKIE_SAMESITE", "none" if COOKIE_SECURE else "lax").lower()
AUTH_DISABLED = os.environ.get("TARGET2_AUTH_DISABLED", "").lower() in {"1", "true", "yes", "on"}
CONTACT_EMAIL = "contact@trilemmaconsulting.com"
MAIL_CONFIG_PATH = Path(
    os.environ.get(
        "TARGET2_NOTIFY_MAIL_CONFIG",
        str(ROOT.parent.parent / "guidaitor" / "ops" / "mail" / "ionos-mail.json"),
    )
)
PUBLIC_PATHS = {
    "/",
    "/target2",
    "/target2/",
    "/login",
    "/auth/google",
    "/oauth2/callback",
    "/oauth2/redirect-uri",
    "/logout",
    "/health",
    "/me",
    "/favicon.ico",
    "/api/auth/login",
    "/api/auth/logout",
    "/api/access/request",
}
PUBLIC_PREFIXES = ("/static/", "/target2/static/")
PASSWORD_CHANGE_PATHS = {"/api/auth/change-password", "/api/auth/logout", "/logout", "/me"}


def load_session_secret() -> str:
    env_secret = os.environ.get("TARGET2_SESSION_SECRET", "").strip()
    if env_secret:
        return env_secret
    if SESSION_SECRET_PATH.exists():
        value = SESSION_SECRET_PATH.read_text(encoding="utf-8").strip()
        if value:
            return value
    SESSION_SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_urlsafe(48)
    SESSION_SECRET_PATH.write_text(value + "\n", encoding="utf-8")
    return value


SESSION_SECRET = load_session_secret()

app = FastAPI(title="TARGET2 Local Bot", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.trilemmaconsulting.com",
        "https://trilemmaconsulting.com",
        "https://target2-api.ethcuela.es",
        "http://127.0.0.1:8791",
        "http://localhost:8791",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
ASK_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("TARGET2_ASK_WORKERS", "2")))
ASK_JOBS: dict[str, dict[str, Any]] = {}
ASK_JOBS_LOCK = threading.Lock()
ASK_JOB_TTL = int(os.environ.get("TARGET2_ASK_JOB_TTL", "3600"))
INDEX_CACHE_MTIME = 0.0


def oauth_configured() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)


def external_request_base_url(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",", 1)[0].strip()
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc)).split(",", 1)[0].strip()
    return f"{proto}://{host}".rstrip("/")


def callback_uri(request: Request) -> str:
    if TARGET2_PUBLIC_BASE_URL:
        return f"{TARGET2_PUBLIC_BASE_URL}/oauth2/callback"
    if GOOGLE_REDIRECT_URI:
        return GOOGLE_REDIRECT_URI
    return f"{external_request_base_url(request)}/oauth2/callback"


def public_auth_url(request: Request) -> str | None:
    if not TARGET2_PUBLIC_BASE_URL:
        return None
    if external_request_base_url(request).lower() == TARGET2_PUBLIC_BASE_URL.lower():
        return None
    query = f"?{request.url.query}" if request.url.query else ""
    return f"{TARGET2_PUBLIC_BASE_URL}{request.url.path}{query}"


def clean_next_path(value: str | None) -> str:
    if not value:
        return "/"
    value = value.strip()
    if not value.startswith("/") or value.startswith("//"):
        return "/"
    if value.startswith(("/login", "/auth/google", "/oauth2/callback", "/access-request")):
        return "/"
    return value


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def sign_payload(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.setdefault("iat", int(time.time()))
    raw = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(SESSION_SECRET.encode("utf-8"), raw, hashlib.sha256).digest()
    return f"{b64url_encode(raw)}.{b64url_encode(sig)}"


def read_signed_payload(token: str | None, max_age: int) -> dict[str, Any] | None:
    if not token or "." not in token:
        return None
    raw_b64, sig_b64 = token.split(".", 1)
    try:
        raw = b64url_decode(raw_b64)
        sig = b64url_decode(sig_b64)
    except Exception:
        return None
    expected = hmac.new(SESSION_SECRET.encode("utf-8"), raw, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    iat = int(payload.get("iat") or 0)
    if not iat or time.time() - iat > max_age:
        return None
    return payload


def read_session(request: Request) -> dict[str, Any] | None:
    token = request.cookies.get(SESSION_COOKIE)
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    payload = read_signed_payload(token, SESSION_MAX_AGE)
    if not payload:
        return None
    email = str(payload.get("email") or "").strip().lower()
    user = approved_user(email)
    if not user:
        return None
    payload["email"] = email
    payload["name"] = user.get("name") or email
    payload["must_change_password"] = bool(user.get("must_change_password"))
    return payload


def set_signed_cookie(response: RedirectResponse | JSONResponse, name: str, payload: dict[str, Any], max_age: int) -> None:
    response.set_cookie(
        name,
        sign_payload(payload),
        max_age=max_age,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
    )


def clear_auth_cookies(response: RedirectResponse | JSONResponse | HTMLResponse) -> None:
    for name in (SESSION_COOKIE, OAUTH_STATE_COOKIE):
        response.delete_cookie(name, httponly=True, secure=COOKIE_SECURE, samesite=COOKIE_SAMESITE)


def wants_json(request: Request) -> bool:
    path = request.url.path
    accept = request.headers.get("accept", "")
    return (
        path.startswith("/api/")
        or path.startswith("/ask")
        or path in {"/manifest", "/health", "/me"}
        or "application/json" in accept
    )


def login_html(request: Request, message: str = "", access_message: str = "") -> str:
    message_html = f"<p class=\"error\">{html.escape(message)}</p>" if message else ""
    access_html = f"<p class=\"ok\">{html.escape(access_message)}</p>" if access_message else ""
    return f"""<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>TARGET2 GPT - Acceso</title>
    <style>
      :root {{ color-scheme: light; --ink:#191917; --muted:#6f6a60; --line:#ddd8cc; --green:#0f8f72; --paper:#fffdf8; --page:#f4f2ec; }}
      * {{ box-sizing: border-box; }}
      body {{ margin:0; min-height:100dvh; display:grid; place-items:center; background:var(--page); color:var(--ink); font-family:ui-sans-serif,"Segoe UI",Aptos,Calibri,sans-serif; padding:22px; }}
      main {{ width:min(460px,100%); background:var(--paper); border:1px solid var(--line); border-radius:8px; box-shadow:0 18px 50px rgba(36,35,31,.14); padding:26px; }}
      .mark {{ width:48px; height:48px; display:grid; place-items:center; border-radius:8px; background:var(--green); color:white; font-weight:900; margin-bottom:18px; }}
      h1 {{ margin:0 0 8px; font-size:24px; letter-spacing:0; }}
      p {{ margin:0 0 14px; line-height:1.45; color:var(--muted); }}
      .google, button {{ width:100%; min-height:46px; border-radius:8px; border:2px solid #bdb5a6; background:white; color:var(--ink); font-weight:900; text-decoration:none; display:grid; place-items:center; cursor:pointer; }}
      .google {{ margin:16px 0 10px; border-color:var(--green); background:#e4f4ee; }}
      .access {{ margin-top:10px; border-color:#245b89; background:#edf4fb; }}
      .google.disabled {{ pointer-events:none; opacity:.55; }}
      .error,.warning {{ color:#9d2e25; font-weight:800; }}
      .ok {{ color:#0a6d58; font-weight:800; }}
      code {{ color:var(--ink); }}
    </style>
  </head>
  <body>
    <main>
      <div class="mark">T</div>
      <h1>TARGET2 GPT</h1>
      <p>Acceso restringido. Primero hay que solicitarlo por email a <strong>{html.escape(CONTACT_EMAIL)}</strong>. Tras la aprobacion se entrega una clave temporal que debe cambiarse en el primer acceso.</p>
      {message_html}
      <a class="google" href="/target2/">Abrir pantalla de acceso</a>
      <a class="google access" href="mailto:{html.escape(CONTACT_EMAIL)}?subject=target2%20GPT%20access%20request">Enviar solicitud por email</a>
      <p>Si el email no esta aprobado localmente, no hay acceso aunque se conozca una clave.</p>
      {access_html}
    </main>
  </body>
</html>"""


def unauthorized_response(request: Request) -> JSONResponse | RedirectResponse:
    if wants_json(request):
        return JSONResponse({"detail": "Authentication required"}, status_code=401)
    return RedirectResponse(f"/target2/?{urlencode({'next': request.url.path})}", status_code=303)


def write_access_request(request: Request, user: dict[str, Any]) -> None:
    record_access_request(
        {
            "ts": int(time.time()),
            "google_email": str(user.get("email") or "").strip().lower(),
            "google_sub": user.get("sub") or "",
            "email_verified": user.get("email_verified"),
            "name": user.get("name") or "",
            "remote": request.client.host if request.client else "",
            "user_agent": request.headers.get("user-agent", ""),
        }
    )


@app.middleware("http")
async def require_google_session(request: Request, call_next):
    path = request.url.path
    if request.method == "OPTIONS":
        return await call_next(request)
    if AUTH_DISABLED:
        request.state.user = {"email": "lan@local", "name": "LAN local"}
        return await call_next(request)
    if path in PUBLIC_PATHS or any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES):
        return await call_next(request)
    session = read_session(request)
    if not session:
        return unauthorized_response(request)
    if session.get("must_change_password") and path not in PASSWORD_CHANGE_PATHS:
        return JSONResponse(
            {
                "detail": "Password change required",
                "code": "password_change_required",
            },
            status_code=403,
        )
    request.state.user = session
    return await call_next(request)


class ChatTurn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    top_k: int = 24
    mode: str = "answer"
    use_codex: bool = True
    model: str = "codex_high"
    language: str = "auto"
    history: list[ChatTurn] = Field(default_factory=list)


class AccessRequestPayload(BaseModel):
    email: str
    name: str = ""
    organization: str = ""
    message: str = ""


class LoginPayload(BaseModel):
    email: str
    password: str


class ChangePasswordPayload(BaseModel):
    current_password: str
    new_password: str


def client_metadata(request: Request) -> dict[str, Any]:
    return {
        "remote": request.client.host if request.client else "",
        "user_agent": request.headers.get("user-agent", ""),
    }


def send_access_request_notification(payload: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("TARGET2_NOTIFY_EMAIL_DISABLED", "").lower() in {"1", "true", "yes", "on"}:
        return {"sent": False, "detail": "notification disabled"}
    if not MAIL_CONFIG_PATH.exists():
        return {"sent": False, "detail": f"mail config not found: {MAIL_CONFIG_PATH}"}
    try:
        import keyring
    except Exception as exc:
        return {"sent": False, "detail": f"keyring unavailable: {exc}"}
    try:
        cfg = json.loads(MAIL_CONFIG_PATH.read_text(encoding="utf-8"))
        account = str(cfg.get("account") or "").strip()
        service = str(cfg.get("password_service") or "clawdbot-ionos").strip()
        password = keyring.get_password(service, account)
        if not account or not password:
            return {"sent": False, "detail": "mail account/password unavailable"}

        msg = EmailMessage()
        msg["From"] = account
        msg["To"] = CONTACT_EMAIL
        msg["Reply-To"] = str(payload.get("email") or account)
        msg["Subject"] = f"TARGET2 GPT access request: {payload.get('email')}"
        lines = [
            "New TARGET2 GPT access request.",
            "",
            f"Email: {payload.get('email')}",
            f"Name: {payload.get('name') or ''}",
            f"Organization: {payload.get('organization') or ''}",
            f"Message: {payload.get('message') or ''}",
            f"Remote: {payload.get('remote') or ''}",
            f"User-Agent: {payload.get('user_agent') or ''}",
            "",
            "Approve locally with:",
            f"python target2gpt\\access_admin.py approve {payload.get('email')} --name \"{payload.get('name') or payload.get('email')}\"",
        ]
        msg.set_content("\n".join(lines))

        smtp_cfg = cfg.get("smtp") or {}
        smtp = smtplib.SMTP(str(smtp_cfg.get("host")), int(smtp_cfg.get("port") or 587), timeout=20)
        try:
            smtp.ehlo()
            if smtp_cfg.get("starttls", False):
                smtp.starttls(context=ssl.create_default_context())
                smtp.ehlo()
            smtp.login(account, password)
            smtp.send_message(msg)
        finally:
            try:
                smtp.quit()
            except Exception:
                pass
        return {"sent": True, "detail": f"notification sent to {CONTACT_EMAIL}"}
    except Exception as exc:
        return {"sent": False, "detail": str(exc)}


def session_response(user: dict[str, Any]) -> JSONResponse:
    session_payload = {
        "email": user.get("email"),
        "name": user.get("name") or user.get("email"),
        "must_change_password": bool(user.get("must_change_password")),
    }
    session_token = sign_payload(session_payload)
    response = JSONResponse(
        {
            "authenticated": True,
            "email": user.get("email"),
            "name": user.get("name") or user.get("email"),
            "must_change_password": bool(user.get("must_change_password")),
            "session_token": session_token,
        }
    )
    set_signed_cookie(
        response,
        SESSION_COOKIE,
        session_payload,
        SESSION_MAX_AGE,
    )
    return response


def ensure_current_rag_index() -> None:
    global INDEX_CACHE_MTIME
    try:
        mtime = INDEX_PATH.stat().st_mtime
    except FileNotFoundError:
        return
    if mtime != INDEX_CACHE_MTIME:
        load_index.cache_clear()
        load_json.cache_clear()
        INDEX_CACHE_MTIME = mtime


@app.post("/api/access/request")
def request_access(payload: AccessRequestPayload, request: Request) -> dict[str, Any]:
    email = normalize_email(payload.email)
    if not valid_email(email):
        raise HTTPException(status_code=400, detail="Valid email is required")
    request_record = {
        "ts": int(time.time()),
        "email": email,
        "name": payload.name.strip(),
        "organization": payload.organization.strip(),
        "message": payload.message.strip(),
        **client_metadata(request),
    }
    record_access_request(request_record)
    notification = send_access_request_notification(request_record)
    return {
        "ok": True,
        "detail": (
            f"Request registered and notification sent to {CONTACT_EMAIL}."
            if notification.get("sent")
            else f"Request registered locally. Notification email was not sent: {notification.get('detail')}"
        ),
        "contact_email": CONTACT_EMAIL,
        "notification_sent": bool(notification.get("sent")),
        "notification_detail": notification.get("detail"),
    }


@app.get("/api/access/requests")
def access_requests() -> dict[str, Any]:
    return {"requests": list_latest_access_requests()}


@app.post("/api/auth/login")
def api_login(payload: LoginPayload) -> JSONResponse:
    email = normalize_email(payload.email)
    user = approved_user(email)
    if not user or not verify_password(payload.password, user.get("password_hash")):
        raise HTTPException(status_code=403, detail="Email is not authorized or the key is invalid")
    return session_response(user)


@app.post("/api/auth/change-password")
def api_change_password(payload: ChangePasswordPayload, request: Request) -> JSONResponse:
    session = read_session(request)
    if not session:
        raise HTTPException(status_code=401, detail="Authentication required")
    user = get_user(str(session.get("email") or ""))
    if not user or user.get("status") != "approved":
        raise HTTPException(status_code=401, detail="Authentication required")
    if not verify_password(payload.current_password, user.get("password_hash")):
        raise HTTPException(status_code=403, detail="Current key is not valid")
    new_password = payload.new_password.strip()
    if len(new_password) < 10:
        raise HTTPException(status_code=400, detail="New key must have at least 10 characters")
    if verify_password(new_password, user.get("password_hash")):
        raise HTTPException(status_code=400, detail="New key must be different from the temporary key")
    user = set_user_password(str(user["email"]), new_password, must_change_password=False)
    return session_response(user)


@app.post("/api/auth/logout")
def api_logout() -> JSONResponse:
    response = JSONResponse({"ok": True})
    clear_auth_cookies(response)
    return response


def history_for_model(history: list[ChatTurn]) -> list[dict[str, str]]:
    return [
        {"role": turn.role, "content": turn.content}
        for turn in history[-10:]
        if turn.role in {"user", "assistant"} and turn.content.strip()
    ]


def contextual_query(question: str, history: list[ChatTurn]) -> str:
    parts = [question.strip()]
    recent = [
        f"{turn.role}: {turn.content.strip()}"
        for turn in history[-10:]
        if turn.role in {"user", "assistant"} and turn.content.strip()
    ]
    if recent:
        parts.append("Recent chat for resolving follow-up references:")
        parts.extend(recent)
    return "\n".join(parts)


@app.get("/login")
def login(request: Request):
    if AUTH_DISABLED:
        return RedirectResponse("/", status_code=303)
    if read_session(request):
        return RedirectResponse(clean_next_path(request.query_params.get("next")), status_code=303)
    return HTMLResponse(login_html(request))


@app.get("/auth/google")
def auth_google(request: Request):
    return HTMLResponse(
        login_html(request, "El acceso por Google esta desactivado. Usa email autorizado y clave temporal."),
        status_code=410,
    )


@app.get("/oauth2/callback")
def oauth_callback(request: Request):
    return HTMLResponse(
        login_html(request, "El acceso por Google esta desactivado. Usa email autorizado y clave temporal."),
        status_code=410,
    )


@app.get("/logout")
def logout() -> RedirectResponse:
    response = RedirectResponse("/target2/", status_code=303)
    clear_auth_cookies(response)
    return response


@app.get("/oauth2/redirect-uri")
def oauth_redirect_uri(request: Request) -> dict[str, Any]:
    return {
        "redirect_uri": callback_uri(request),
        "public_base_url": TARGET2_PUBLIC_BASE_URL or None,
        "request_base_url": external_request_base_url(request),
        "google_client_type": GOOGLE_CLIENT_TYPE or None,
        "access_requests_path": str(ACCESS_REQUESTS_PATH),
        "auth_disabled": AUTH_DISABLED,
    }


@app.get("/me")
def me(request: Request) -> dict[str, Any]:
    session = read_session(request)
    if not session:
        return {
            "authenticated": False,
            "contact_email": CONTACT_EMAIL,
        }
    return {
        "authenticated": True,
        "email": session.get("email"),
        "name": session.get("name"),
        "must_change_password": bool(session.get("must_change_password")),
        "contact_email": CONTACT_EMAIL,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": INDEX_PATH.exists(), "index": str(INDEX_PATH)}


@app.get("/manifest")
def manifest() -> dict[str, Any]:
    ensure_current_rag_index()
    path = ROOT / "data" / "processed" / "manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run python target2_ingest.py first")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def error_answer(req: AskRequest, exc: Exception) -> dict[str, Any]:
    return {
        "question": req.question,
        "answer": (
            "No he podido completar esta consulta, pero la web sigue viva. "
            "Prueba a reformularla con el termino TARGET2, T2, CLM o RTGS concreto, o cambia a modo Contexto para ver la evidencia recuperada."
        ),
        "citations": [],
        "confidence": "low",
        "generated_by": "error_guard",
        "error": str(exc),
    }


def answer_request(req: AskRequest) -> dict[str, Any]:
    try:
        ensure_current_rag_index()
        search_query = contextual_query(req.question, req.history)
        if req.mode == "context":
            index = load_index()
            retrieval_k = max(req.top_k, 32)
            hits = retrieve(index, search_query, top_k=retrieval_k)
            hits = augment_hits(index, search_query, hits)
            hits = expand_neighbor_hits(index, hits, max_neighbors=1, max_total=min(retrieval_k + 16, MAX_CONTEXT_HITS))
            hits = rerank_context_hits(search_query, hits, max_total=min(retrieval_k + 16, MAX_CONTEXT_HITS))
            return build_codex_context(search_query, hits, max_hits=req.top_k)
        return answer_question(
            req.question,
            top_k=req.top_k,
            language=req.language,
            generate=req.use_codex and req.model != "local_rag",
            model_preset=req.model,
            retrieval_query=search_query,
            chat_history=history_for_model(req.history),
        )
    except Exception as exc:
        return error_answer(req, exc)


def cleanup_ask_jobs() -> None:
    cutoff = time.time() - ASK_JOB_TTL
    with ASK_JOBS_LOCK:
        for job_id, job in list(ASK_JOBS.items()):
            if job.get("status") in {"done", "error", "cancelled"} and float(job.get("updated_at") or 0) < cutoff:
                ASK_JOBS.pop(job_id, None)


def set_ask_job(job_id: str, **updates: Any) -> None:
    with ASK_JOBS_LOCK:
        job = ASK_JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = time.time()


def ask_job_snapshot(job_id: str) -> dict[str, Any]:
    with ASK_JOBS_LOCK:
        job = ASK_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Ask job not found")
        return {key: value for key, value in job.items() if key != "future"}


def run_ask_job(job_id: str, req: AskRequest) -> None:
    set_ask_job(job_id, status="running", stage="retrieving")
    try:
        result = answer_request(req)
        with ASK_JOBS_LOCK:
            job = ASK_JOBS.get(job_id)
            if not job:
                return
            if job.get("status") == "cancelled":
                return
            job.update(status="done", stage="done", result=result, updated_at=time.time())
    except Exception as exc:
        set_ask_job(job_id, status="error", stage="error", error=str(exc), result=error_answer(req, exc))


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")
    return answer_request(req)


@app.post("/ask/jobs")
def start_ask_job(req: AskRequest) -> dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")
    cleanup_ask_jobs()
    job_id = uuid.uuid4().hex
    now = time.time()
    with ASK_JOBS_LOCK:
        ASK_JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "question": req.question,
            "created_at": now,
            "updated_at": now,
        }
    future = ASK_EXECUTOR.submit(run_ask_job, job_id, req)
    with ASK_JOBS_LOCK:
        if job_id in ASK_JOBS:
            ASK_JOBS[job_id]["future"] = future
    return {"job_id": job_id, "status": "queued"}


@app.get("/ask/jobs/{job_id}")
def get_ask_job(job_id: str) -> dict[str, Any]:
    return ask_job_snapshot(job_id)


@app.delete("/ask/jobs/{job_id}")
def cancel_ask_job(job_id: str) -> dict[str, Any]:
    with ASK_JOBS_LOCK:
        job = ASK_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Ask job not found")
        future = job.get("future")
        if future:
            future.cancel()
        job.update(status="cancelled", stage="cancelled", updated_at=time.time())
    return ask_job_snapshot(job_id)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(PUBLIC / "index.html")


@app.get("/target2")
@app.get("/target2/")
def target2_home() -> FileResponse:
    return FileResponse(PUBLIC / "index.html")


app.mount("/target2/static", StaticFiles(directory=PUBLIC), name="target2-static")
app.mount("/static", StaticFiles(directory=PUBLIC), name="static")


if __name__ == "__main__":
    host = os.environ.get("TARGET2_WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("TARGET2_WEB_PORT", "8791"))
    uvicorn.run("target2_web:app", host=host, port=port, reload=False)


