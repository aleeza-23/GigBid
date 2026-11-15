"""Sign-in, sign-out and who-am-I.

The session is a signed, expiring token in an HttpOnly cookie, and also returned in the
body so the generated TypeScript client and curl both work without a cookie jar.

HttpOnly matters: the token is a bearer credential for an API that can spend the user's
Connects, so JavaScript should not be able to read it.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Request, Response, status

from apps.api.deps import (
    SESSION_COOKIE,
    SESSION_TTL_SEC,
    CurrentUser,
    authenticate,
    sign_session,
)
from apps.api.schemas.models import LoginRequest, SessionResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=SessionResponse)
async def login(body: LoginRequest, response: Response) -> SessionResponse:
    """Sign in. The reviewer account is seeded from the environment."""
    user = authenticate(body.username, body.password)
    if user is None:
        # One message for both wrong-username and wrong-password, so the response does
        # not confirm which accounts exist.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "incorrect username or password")

    token = sign_session(user["user_id"])
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        max_age=SESSION_TTL_SEC,
        samesite="lax",
        # Secure only in production: a local http:// dev server would silently drop it.
        secure=os.getenv("APP_ENV", "development") == "production",
    )
    return SessionResponse(
        user_id=user["user_id"],
        display_name=user.get("display_name") or user["user_id"],
        persona=user.get("persona") or "ai_ml_engineer",
        token=token,
    )


@router.post("/logout")
async def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie(SESSION_COOKIE)
    return {"signed_out": True}


@router.get("/me", response_model=SessionResponse)
async def me(request: Request, user: dict = CurrentUser) -> SessionResponse:
    """Who the current session belongs to. The UI calls this on load."""
    token = request.cookies.get(SESSION_COOKIE) or ""
    return SessionResponse(
        user_id=user["user_id"],
        display_name=user["display_name"],
        persona=user["persona"],
        token=token,
    )
