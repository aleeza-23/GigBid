"""The one way GigBid talks to a marketplace (R1).

Every marketplace call in the system goes through MarketplacePort. Two adapters sit behind
it, GigSim and live Upwork, chosen by config. The graph never learns which one it is
talking to, which is what makes switching to live mode a configuration change rather than
a code change.

This module is the seam. It holds the interface, the error taxonomy and the factory; the
adapters themselves live under `mcp_servers/marketplace/adapters/` next to the MCP server
that exposes them as tools. The dependency runs one way, port -> adapters -> base, so
there is no cycle and `base.py` can be imported by anything.

Layering, since it is easy to get backwards:

    nodes -> MCP marketplace server -> MarketplacePort -> GigSim | Upwork live

Nodes never import an adapter. They call MCP tools; the server behind those tools calls
the port; the port picks the adapter. A6's per-agent tool allowlists are enforced at the
MCP boundary, which only works if nothing reaches around it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mcp_servers.marketplace.adapters.base import (
    MarketplaceError,
    MarketplacePort,
    MarketplaceRejected,
    MarketplaceUnavailable,
    new_idempotency_key,
)
from mcp_servers.marketplace.adapters.gigsim import GigSimAdapter
from mcp_servers.marketplace.adapters.upwork_live import UpworkLiveAdapter

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

__all__ = [
    "CONFIG_PATH",
    "GigSimAdapter",
    "MarketplaceError",
    "MarketplacePort",
    "MarketplaceRejected",
    "MarketplaceUnavailable",
    "UpworkLiveAdapter",
    "get_marketplace",
    "load_config",
    "new_idempotency_key",
]


def load_config(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or CONFIG_PATH).read_text(encoding="utf-8"))


def get_marketplace(config: dict[str, Any] | None = None, **kwargs: Any) -> MarketplacePort:
    """Build the adapter named in config. The only place a mode string is read.

    Keeping this the single reader of `marketplace.mode` is what makes the switch
    auditable: grep for the string and there is one hit.
    """
    cfg = config or load_config()
    mode = (cfg.get("marketplace") or {}).get("mode", "gigsim")
    if mode == "gigsim":
        return GigSimAdapter(**kwargs)
    if mode == "upwork_live":
        return UpworkLiveAdapter(**kwargs)
    raise ValueError(f"unknown marketplace mode {mode!r}: expected 'gigsim' or 'upwork_live'")
