"""Inicio de sesión del psiquiatra y datos del usuario autenticado."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ....core.dependencies import Container, get_container
from ....core.security import CurrentUser, create_token, current_user, verify_password
from ....domain.schemas import LoginRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, c: Container = Depends(get_container)):
    auth = c.settings.auth
    u = c.db.get_user(body.username)
    if u is None or not verify_password(body.password, u.password_hash):
        raise HTTPException(401, "credenciales inválidas")
    if u.role not in auth.allowed_roles:
        raise HTTPException(403, "rol no autorizado")
    token = create_token(u.id, u.username, u.role, auth.secret_key, auth.algorithm, auth.token_expire_minutes)
    return {"access_token": token, "token_type": "bearer", "user": {"id": u.id, "username": u.username, "role": u.role, "full_name": u.full_name}}


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser = Depends(current_user), c: Container = Depends(get_container)):
    u = c.db.get_user(user.username)
    return {"id": u.id, "username": u.username, "role": u.role, "full_name": u.full_name}
