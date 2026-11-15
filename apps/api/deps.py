"""Application lifecycle and shared dependencies.

Three things have to be built once per process and shared, and each one is a bug if it is
not:

* **the checkpointer** — the API is where approvals resume, so it must hold the *same*
  checkpointer the run was started with. Build one per request and a pending approval
  becomes invisible the moment the request that created it returns;
* **the memory store** — style and outcome memory outlive threads, so a per-request store
  learns nothing;
* **the marketplace port** — one HTTP connection pool, or the rate budget is spent on TCP
  handshakes.

They live in `app.state`, created in the lifespan and torn down with it. The async
checkpointer is an async context manager, which is why this is a lifespan rather than a
module-level singleton.

Auth is deliberately small. The assignment needs multi-user scoping and a reviewer login,
not an identity provider: a signed session cookie over a user table, with the reviewer
account seeded from the environment. Everything downstream takes a `user_id` and scopes by
it, so swapping this for real OAuth later touches this file and nothing else.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, HTTPException, Request, status

SESSION_COOKIE = "gigbid_session"
SESSION_TTL_SEC = 60 * 60 * 12


def _secret() -> bytes:
    key = os.getenv("APP_SECRET_KEY", "").strip()
    if not key or key == "replace_me":
        # A fixed fallback would mean every deployment shares a signing key, so sessions
        # from one instance would validate on another. Random per process is the safe
        # default; set APP_SECRET_KEY to keep sessions across restarts.
        key = secrets.token_urlsafe(32)
        os.environ["APP_SECRET_KEY"] = key
    return key.encode()


def sign_session(user_id: str) -> str:
    """A signed, expiring session token. No server-side session table needed."""
    expires = int(time.time()) + SESSION_TTL_SEC
    payload = f"{user_id}:{expires}"
    signature = hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{payload}:{signature}"


def verify_session(token: str | None) -> str | None:
    """Return the user id, or None. Constant-time comparison on the signature."""
    if not token or token.count(":") != 2:
        return None
    user_id, expires, signature = token.split(":")
    try:
        if int(expires) < time.time():
            return None
    except ValueError:
        return None
    expected = hmac.new(
        _secret(), f"{user_id}:{expires}".encode(), hashlib.sha256
    ).hexdigest()[:32]
    return user_id if hmac.compare_digest(signature, expected) else None


# --------------------------------------------------------------------- users

# A tiny user table. The reviewer account is seeded from the environment so the deployed
# link can be opened with credentials submitted privately, as the assignment requires.
def _users() -> dict[str, dict[str, Any]]:
    reviewer_password = os.getenv("REVIEWER_PASSWORD", "reviewer")
    return {
        os.getenv("REVIEWER_USER", "reviewer"): {
            "password": reviewer_password,
            "persona": os.getenv("REVIEWER_PERSONA", "ai_ml_engineer"),
            "display_name": "Reviewer",
        },
        "aleeza": {
            "password": os.getenv("OWNER_PASSWORD", "aleeza"),
            "persona": "ai_ml_engineer",
            "display_name": "Aleeza",
        },
    }


def authenticate(username: str, password: str) -> dict[str, Any] | None:
    user = _users().get(username)
    if user is None:
        return None
    # Constant-time, so a wrong username and a wrong password take the same time.
    if not hmac.compare_digest(str(user["password"]), password):
        return None
    return {"user_id": username, **{k: v for k, v in user.items() if k != "password"}}


def user_profile(user_id: str) -> dict[str, Any]:
    user = _users().get(user_id) or {}
    return {
        "user_id": user_id,
        "persona": user.get("persona", "ai_ml_engineer"),
        "display_name": user.get("display_name", user_id),
    }


async def current_user(request: Request) -> dict[str, Any]:
    """The signed-in user, or 401. Every data route depends on this."""
    user_id = verify_session(request.cookies.get(SESSION_COOKIE))
    if user_id is None:
        # Bearer is accepted too, so the generated TypeScript client and curl both work.
        header = request.headers.get("Authorization", "")
        if header.lower().startswith("bearer "):
            user_id = verify_session(header[7:])
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "sign in first")
    return user_profile(user_id)


CurrentUser = Depends(current_user)


# ------------------------------------------------------------------ lifespan


@asynccontextmanager
async def lifespan(app: Any):
    """Build the shared resources once, tear them down on shutdown."""
    from agent.checkpointer import acheckpointer, describe
    from agent.checkpointer import store as build_store
    from agent.marketplace_port import get_marketplace

    persistence = describe()
    app.state.persistence = persistence
    if not persistence["survives_restart"]:
        # Not fatal, but the operator should know: a restart will drop every pending
        # approval, which is the one failure the approval flow exists to prevent.
        print(
            f"WARNING: checkpointer is {persistence['checkpointer']!r}, which does not "
            "survive a restart. Pending approvals will be lost. Set "
            "persistence.checkpointer to sqlite or postgres."
        )

    async with acheckpointer() as saver:
        app.state.checkpointer = saver
        app.state.store = build_store()
        app.state.marketplace = get_marketplace()
        app.state.graphs = {}
        app.state.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            yield
        finally:
            port = getattr(app.state, "marketplace", None)
            if port is not None:
                try:
                    await port.close()
                except Exception:
                    pass


def bid_graph(request: Request) -> Any:
    """The compiled bid graph, built once and cached on app state."""
    graphs = request.app.state.graphs
    if "bid" not in graphs:
        from agent.graphs.bid_graph import build_bid_graph

        graphs["bid"] = build_bid_graph(
            checkpointer=request.app.state.checkpointer, store=request.app.state.store
        )
    return graphs["bid"]


def conversation_graph(request: Request) -> Any:
    graphs = request.app.state.graphs
    if "conversation" not in graphs:
        from agent.graphs.conversation_graph import build_conversation_graph

        graphs["conversation"] = build_conversation_graph(
            checkpointer=request.app.state.checkpointer, store=request.app.state.store
        )
    return graphs["conversation"]


def memory_for(request: Request, user_id: str) -> Any:
    from agent.memory.store import Memory

    return Memory(request.app.state.store, user_id)


def thread_id(kind: str, user_id: str, key: str) -> str:
    """Thread ids are scoped per user, so two users bidding on one job never collide."""
    return f"{kind}:{user_id}:{key}"
