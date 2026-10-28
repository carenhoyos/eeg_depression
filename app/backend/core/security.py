"""Autenticación: contraseñas con PBKDF2 (stdlib) y tokens JWT (PyJWT). Rol único por ahora: psychiatrist."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .dependencies import Container, get_container

_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str, iterations: int) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt_b64, dk_b64 = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), base64.b64decode(salt_b64), int(iters))
        return hmac.compare_digest(dk, base64.b64decode(dk_b64))
    except (ValueError, TypeError):
        return False


def create_token(user_id: int, username: str, role: str, secret: str, algorithm: str, expires_minutes: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "username": username, "role": role, "iat": now, "exp": now + timedelta(minutes=expires_minutes)}
    return jwt.encode(payload, secret, algorithm=algorithm)


def decode_token(token: str, secret: str, algorithm: str) -> dict[str, Any]:
    return jwt.decode(token, secret, algorithms=[algorithm])


class CurrentUser:
    def __init__(self, id: int, username: str, role: str):
        self.id, self.username, self.role = id, username, role


def current_user(request: Request, creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
                 c: Container = Depends(get_container)) -> CurrentUser:
    auth = c.settings.auth
    if not auth.enabled:
        u = c.db.get_user(auth.bootstrap_username)
        return CurrentUser(u.id, u.username, u.role)
    if creds is None:
        raise HTTPException(401, "token requerido")
    try:
        payload = decode_token(creds.credentials, auth.secret_key, auth.algorithm)
    except jwt.PyJWTError:
        raise HTTPException(401, "token inválido o expirado")
    if payload.get("role") not in auth.allowed_roles:
        raise HTTPException(403, "rol no autorizado")
    return CurrentUser(int(payload["sub"]), payload["username"], payload["role"])
