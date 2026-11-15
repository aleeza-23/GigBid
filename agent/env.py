"""Load .env once, as early as possible.

Without this, `os.getenv("GROQ_API_KEY")` returns None however carefully .env was filled
in, and the whole system silently falls back to offline mode. That failure is quiet and
confusing, so loading happens here and every entry point imports it.

`override=False` is deliberate: a variable already set in the real environment wins over
.env. That is what makes `GIGSIM_ERROR_RATE=0.05 make demo` work, and it is how the
deployed stack injects secrets without a .env file existing at all.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"

_loaded = False


def load_env(path: Path | None = None) -> bool:
    """Load .env into the process environment. Idempotent; returns whether a file was read."""
    global _loaded
    if _loaded:
        return True

    target = path or ENV_PATH
    if not target.exists():
        _loaded = True
        return False

    try:
        from dotenv import load_dotenv

        load_dotenv(target, override=False)
    except ImportError:
        # Minimal fallback so a missing python-dotenv does not take the system offline.
        for raw in target.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

    _loaded = True
    return True


def first_env(names: str | list[str], default: str = "") -> tuple[str, str]:
    """The first of `names` that is set, as (value, name_it_came_from).

    Model providers are mostly OpenAI-compatible, so people put a Groq key in
    OPENAI_API_KEY and a Together key in OPENROUTER_API_KEY all the time. Accepting a
    list of candidates turns a silent offline fallback into a working call, and returning
    the name lets the caller say which variable it actually used.
    """
    load_env()
    candidates = [names] if isinstance(names, str) else list(names)
    for name in candidates:
        value = os.getenv(name, "").strip()
        if value:
            return value, name
    return default, ""


# Loading at import means `import agent.env` is enough; no caller has to remember to call
# load_env() first.
load_env()
