"""Runtime settings for GigSim, all overridable by environment variable.

Everything that makes the simulator harder or easier lives here so that an evaluation
run can be reproduced exactly: the random seed, the error rate, the rate limit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = REPO_ROOT / "data" / "gigsim_seed"
DERIVED_DIR = REPO_ROOT / "data" / "derived"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class Settings:
    """Simulator configuration.

    api_token          bearer token every request outside /health must present
    rate_limit_per_min requests allowed per token per minute before 429
    error_rate         probability that a read returns a synthetic 5xx
    seed               makes error injection and client decisions reproducible
    client_engine      "rule" (deterministic, no API key) or "llm"
    connects_enforced  when False, proposals never fail for lack of Connects
    """

    api_token: str = field(default_factory=lambda: os.getenv("GIGSIM_API_TOKEN", "gigsim-dev-token"))
    rate_limit_per_min: int = field(default_factory=lambda: _env_int("GIGSIM_RATE_LIMIT_PER_MIN", 30))
    error_rate: float = field(default_factory=lambda: _env_float("GIGSIM_ERROR_RATE", 0.0))
    seed: int = field(default_factory=lambda: _env_int("GIGSIM_SEED", 42))
    client_engine: str = field(default_factory=lambda: os.getenv("GIGSIM_CLIENT_ENGINE", "rule"))
    connects_enforced: bool = field(
        default_factory=lambda: os.getenv("GIGSIM_CONNECTS_ENFORCED", "1") != "0"
    )
    seed_dir: Path = SEED_DIR
    derived_dir: Path = DERIVED_DIR


settings = Settings()
