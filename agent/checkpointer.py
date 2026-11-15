"""Checkpointer and store selection (C6).

C6 requires a Postgres checkpointer and this demonstration: kill the API while a proposal
waits for approval, restart, approve from the UI, and the proposal is submitted exactly
once. That needs persistence the process does not own.

Three backends behind one factory, chosen by `persistence.checkpointer` in config:

| backend  | survives restart | needs |
|----------|------------------|-------|
| postgres | yes              | langgraph-checkpoint-postgres, a database |
| sqlite   | yes              | langgraph-checkpoint-sqlite |
| memory   | **no**           | nothing |

`memory` is the test and CI backend. It is never silently substituted for a configured
durable one: a missing driver raises with the install command, because a deployment that
quietly falls back to memory loses every pending approval on the next restart and looks
fine until it does.

Exactly-once has two independent halves, and both are needed:

* the **checkpointer** restores the thread so the approval is still pending after a crash;
* the **idempotency store** (`mcp_servers/_common/idempotency.py`) stops the write landing
  twice if the crash happened mid-submit.

A checkpointer alone would replay the submit node after a crash. An idempotency key alone
would lose the pending approval. The demonstration C6 asks for needs both.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

import yaml

from agent.serde import serializer

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_SQLITE = Path(__file__).resolve().parents[1] / "data" / "audit" / "checkpoints.sqlite3"


class PersistenceError(RuntimeError):
    """A configured backend that cannot be built."""


def load_config(path: Path | None = None) -> dict[str, Any]:
    try:
        return yaml.safe_load((path or CONFIG_PATH).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config if config is not None else load_config()
    return cfg.get("persistence") or {}


def backend_name(config: dict[str, Any] | None = None) -> str:
    return os.getenv("CHECKPOINTER") or _settings(config).get("checkpointer", "memory")


def database_url() -> str:
    return os.getenv("DATABASE_URL", "")


@contextmanager
def checkpointer(config: dict[str, Any] | None = None, backend: str | None = None):
    """Yield a checkpointer. A context manager because Postgres and SQLite own connections.

    Usage:

        with checkpointer() as saver:
            graph = build_bid_graph(checkpointer=saver)
    """
    name = (backend or backend_name(config)).lower()

    if name == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise PersistenceError(
                "persistence.checkpointer is 'postgres' but the driver is missing:\n"
                "  pip install 'langgraph-checkpoint-postgres' psycopg[binary]\n"
                "Refusing to fall back to memory: a deployment that does would lose every "
                "pending approval on restart."
            ) from exc
        url = database_url()
        if not url:
            raise PersistenceError("DATABASE_URL is not set, so Postgres cannot be reached.")
        # psycopg wants a plain postgresql:// URL; strip any SQLAlchemy driver suffix.
        url = url.replace("postgresql+psycopg://", "postgresql://").replace(
            "postgresql+asyncpg://", "postgresql://"
        )
        with PostgresSaver.from_conn_string(url, serde=serializer()) as saver:
            saver.setup()
            yield saver
        return

    if name == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise PersistenceError(
                "persistence.checkpointer is 'sqlite' but the driver is missing:\n"
                "  pip install langgraph-checkpoint-sqlite"
            ) from exc
        path = Path(os.getenv("CHECKPOINT_DB", str(DEFAULT_SQLITE)))
        path.parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(path)) as saver:
            # SqliteSaver takes its serde positionally in some versions; set it after.
            saver.serde = serializer()
            yield saver
        return

    if name == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        # The same serializer in memory as on disk, so a type that would fail to come
        # back out of Postgres fails in the tests too rather than only in production.
        yield InMemorySaver(serde=serializer())
        return

    raise PersistenceError(
        f"unknown checkpointer {name!r}: expected postgres, sqlite or memory"
    )


@asynccontextmanager
async def acheckpointer(config: dict[str, Any] | None = None, backend: str | None = None):
    """The async counterpart, and the one the graphs actually need.

    Both graphs are async (`ainvoke`), and the durable savers are split: `SqliteSaver` and
    `PostgresSaver` raise `NotImplementedError` on every async method and tell you to use
    the `Async*` variant. `InMemorySaver` supports both, which is why the sync factory
    above appeared to work right up until a durable backend was configured -- the exact
    moment it matters.

        async with acheckpointer() as saver:
            graph = build_bid_graph(checkpointer=saver)
            await graph.ainvoke(state, config)
    """
    name = (backend or backend_name(config)).lower()

    if name == "postgres":
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as exc:
            raise PersistenceError(
                "persistence.checkpointer is 'postgres' but the async driver is missing:\n"
                "  pip install 'langgraph-checkpoint-postgres' psycopg[binary]"
            ) from exc
        url = database_url()
        if not url:
            raise PersistenceError("DATABASE_URL is not set, so Postgres cannot be reached.")
        url = url.replace("postgresql+psycopg://", "postgresql://").replace(
            "postgresql+asyncpg://", "postgresql://"
        )
        async with AsyncPostgresSaver.from_conn_string(url) as saver:
            await saver.setup()
            saver.serde = serializer()
            yield saver
        return

    if name == "sqlite":
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as exc:
            raise PersistenceError(
                "persistence.checkpointer is 'sqlite' but the async driver is missing:\n"
                "  pip install langgraph-checkpoint-sqlite aiosqlite"
            ) from exc
        path = Path(os.getenv("CHECKPOINT_DB", str(DEFAULT_SQLITE)))
        path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
            saver.serde = serializer()
            yield saver
        return

    if name == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver(serde=serializer())
        return

    raise PersistenceError(f"unknown checkpointer {name!r}: expected postgres, sqlite or memory")


def store(config: dict[str, Any] | None = None, backend: str | None = None) -> Any:
    """The long-term memory store (C2).

    Separate from the checkpointer on purpose. A checkpoint is one thread's state and dies
    with the thread; the store holds promoted knowledge that outlives every thread, which
    is why style memory learned on one job applies to the next.
    """
    # `STORE` for the same reason `CHECKPOINTER` exists: a deployment picks its backend
    # from the environment, not from a file baked into the image. Without it the deployed
    # API fell back to the in-memory store, so promoted style and outcome memory (C7) was
    # lost every time a sleeping free instance restarted, while thread checkpoints were
    # durable in Postgres and looked fine.
    name = (backend or os.getenv("STORE") or _settings(config).get("store", "memory")).lower()

    if name == "postgres":
        try:
            from langgraph.store.postgres import PostgresStore
        except ImportError as exc:
            raise PersistenceError(
                "persistence.store is 'postgres' but the driver is missing:\n"
                "  pip install 'langgraph-checkpoint-postgres' psycopg[binary]"
            ) from exc
        url = database_url().replace("postgresql+psycopg://", "postgresql://")
        if not url:
            raise PersistenceError("DATABASE_URL is not set.")
        instance = PostgresStore.from_conn_string(url)
        instance.setup()
        return instance

    from langgraph.store.memory import InMemoryStore

    return InMemoryStore()


def describe(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """What persistence is configured, and whether it actually survives a restart.

    Printed at startup and shown on the settings screen, because "durable" is the kind of
    claim that should be visible rather than assumed.
    """
    name = backend_name(config)
    durable = name in ("postgres", "sqlite")
    available = True
    detail = ""
    if name == "postgres":
        try:
            import langgraph.checkpoint.postgres

            available = bool(database_url())
            detail = "" if available else "DATABASE_URL is not set"
        except ImportError:
            available, detail = False, "langgraph-checkpoint-postgres is not installed"
    elif name == "sqlite":
        try:
            import langgraph.checkpoint.sqlite  # noqa: F401
        except ImportError:
            available, detail = False, "langgraph-checkpoint-sqlite is not installed"

    return {
        "checkpointer": name,
        # Read the same way `store()` resolves it, or the settings screen reports a
        # backend the process is not using.
        "store": (os.getenv("STORE") or _settings(config).get("store", "memory")).lower(),
        "survives_restart": durable,
        "available": available,
        "detail": detail,
        "idempotency_db": os.getenv("MCP_IDEMPOTENCY_DB", "data/audit/idempotency.sqlite3"),
    }
