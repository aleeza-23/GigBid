"""MultiServerMCPClient setup and per-agent tool allowlists (A6).

Two jobs:

1. **Connect** to the five servers GigBid uses: our Portfolio, Marketplace and Workspace,
   plus the two third-party ones (official Upwork, GitHub). Transport and URL come from
   config, so the same graph runs against stdio servers locally and HTTP servers in the
   deployed stack.

2. **Filter** tools by name, per agent. This is the part that matters. The Compliance
   Guard holds no tools at all; the Proposal Writer can read the portfolio but cannot
   touch the marketplace; only the Client Liaison can send a message, and only after a
   human approved it. A Writer that *could* call submit_proposal is one prompt injection
   away from doing it, so the capability is removed rather than discouraged.

The allowlist is data, not code: ALLOWLISTS below mirrors the table in the assignment, and
`tools_for()` is the only way a node gets tools. Nothing stops a node from importing the
marketplace port directly except review, which is why the port is not re-exported here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

# Per-agent tool allowlists, keyed by the role names in the assignment's agent table.
#
# Read the second column as "what this agent could do if a job post talked it into
# trying". Scout can list jobs. Writer can read a portfolio. Only Client Liaison reaches
# a write tool, and that write is gated by an interrupt upstream.
ALLOWLISTS: dict[str, tuple[str, ...]] = {
    # Marketplace (read) - finds and queues jobs.
    "scout": ("search_jobs", "get_job", "marketplace_health"),
    # No tools. The Job Analyst reads the job text already in state and extracts a
    # JobSpec; giving it search tools would invite it to wander.
    "analyst": (),
    # Marketplace (read) - needs client history to judge a client, nothing more.
    "risk_assessor": ("get_client_history", "get_job"),
    # Portfolio (read) - maps requirements to evidence.
    "fit_matcher": ("search_evidence", "get_case_study", "list_skills", "portfolio_health"),
    # Marketplace (read) - market stats, client spend, and the Connects maths.
    "pricing": ("get_rate_stats", "get_client_history", "get_connects_balance"),
    # Portfolio (read) only. Deliberately cannot submit anything.
    "writer": ("search_evidence", "get_case_study", "list_skills"),
    # Portfolio (read) - re-checks that each claim's citation holds up.
    "critic": ("search_evidence", "get_case_study"),
    # No tools. The Compliance Guard judges text against the policy pack; a tool call
    # would only give it a way to be talked into something.
    "compliance": (),
    # Marketplace read plus the two gated writes, and the workspace for scheduling.
    "client_liaison": (
        "list_threads",
        "list_messages",
        "send_message",
        "get_job",
        "free_interview_slots",
        "create_interview_hold",
        "release_interview_hold",
        "list_interview_holds",
        "create_email_draft",
    ),
    # The approval node is what actually spends Connects, after interrupt() returned an
    # approval. Separated from the Writer so the capability sits with the human's decision
    # rather than with the model that drafted the text.
    "approval": ("create_proposal_draft", "submit_proposal", "get_connects_balance"),
    # Used by the R&D question A baseline: one ReAct agent holding everything.
    "all": ("*",),
}

# Tools that spend money or leave the system. Listed explicitly so a test can assert no
# drafting agent holds one, which is a cheaper guarantee than reviewing the table by eye.
WRITE_TOOLS = frozenset({"submit_proposal", "send_message", "create_interview_hold"})

DRAFTING_AGENTS = ("analyst", "risk_assessor", "fit_matcher", "pricing", "writer", "critic", "compliance")


def load_config(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or CONFIG_PATH).read_text(encoding="utf-8"))


def server_config(config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Connection settings for every MCP server, in MultiServerMCPClient's format.

    stdio for our own servers by default: no ports to allocate, no auth to configure, and
    the server dies with the process. The deployed stack sets MCP_TRANSPORT=http and the
    URLs below so the API and worker share one set of servers.
    """
    cfg = config or load_config()
    mcp_cfg = cfg.get("mcp") or {}
    transport = os.getenv("MCP_TRANSPORT", mcp_cfg.get("transport", "stdio"))

    def local(module: str, port: int, name: str) -> dict[str, Any]:
        if transport == "http":
            token = os.getenv("MCP_BEARER_TOKEN", "mcp-dev-token")
            return {
                "transport": "streamable_http",
                "url": os.getenv(f"{name.upper()}_MCP_URL", f"http://127.0.0.1:{port}/mcp"),
                "headers": {"Authorization": f"Bearer {token}"},
            }
        return {
            "transport": "stdio",
            "command": os.getenv("PYTHON_BIN", "python"),
            "args": ["-m", module],
        }

    servers: dict[str, dict[str, Any]] = {
        "portfolio": local("mcp_servers.portfolio.server", 8701, "portfolio"),
        "marketplace": local("mcp_servers.marketplace.server", 8702, "marketplace"),
        "workspace": local("mcp_servers.workspace.server", 8703, "workspace"),
    }

    # Third-party servers (A5). Both are opt-in: absent credentials means the server is
    # simply not registered, rather than every run failing on a connection error.
    if os.getenv("GITHUB_MCP_TOKEN"):
        servers["github"] = {
            "transport": "streamable_http",
            "url": os.getenv("GITHUB_MCP_URL", "https://api.githubcopilot.com/mcp/"),
            "headers": {"Authorization": f"Bearer {os.getenv('GITHUB_MCP_TOKEN')}"},
        }
    if os.getenv("UPWORK_ACCESS_TOKEN"):
        servers["upwork"] = {
            "transport": "streamable_http",
            "url": os.getenv("UPWORK_MCP_URL", "https://mcp.upwork.com/mcp"),
            "headers": {"Authorization": f"Bearer {os.getenv('UPWORK_ACCESS_TOKEN')}"},
        }
    return servers


class GigBidMCPClient:
    """Connects to the MCP servers and hands each agent only its own tools."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.servers = server_config(self.config)
        self._client: Any = None
        self._all_tools: list[Any] | None = None
        self._unavailable: dict[str, str] = {}

    async def client(self) -> Any:
        if self._client is None:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            self._client = MultiServerMCPClient(self.servers)
        return self._client

    async def all_tools(self, refresh: bool = False) -> list[Any]:
        """Every tool from every reachable server.

        A server that cannot be reached is recorded in `unavailable` and skipped rather
        than failing the lot. A6 asks for a clear degraded mode: losing the marketplace
        should leave the Fit Matcher working and the UI saying what is down.
        """
        if self._all_tools is not None and not refresh:
            return self._all_tools

        client = await self.client()
        collected: list[Any] = []
        self._unavailable = {}
        for name in self.servers:
            try:
                collected.extend(await client.get_tools(server_name=name))
            except Exception as exc:
                self._unavailable[name] = f"{type(exc).__name__}: {exc}"
        self._all_tools = collected
        return collected

    @property
    def unavailable(self) -> dict[str, str]:
        """Servers that failed on the last `all_tools` call, for the degraded banner."""
        return dict(self._unavailable)

    async def tools_for(self, agent: str) -> list[Any]:
        """The tools one agent may use. The only way a node should get tools."""
        allowed = ALLOWLISTS.get(agent)
        if allowed is None:
            raise KeyError(
                f"no tool allowlist for agent {agent!r}. Add one to ALLOWLISTS rather than "
                "passing unfiltered tools: an agent with every tool is the baseline we are "
                "measuring against, not the design."
            )
        tools = await self.all_tools()
        if allowed == ("*",):
            return tools
        by_name = {getattr(t, "name", ""): t for t in tools}
        return [by_name[n] for n in allowed if n in by_name]

    async def close(self) -> None:
        client, self._client, self._all_tools = self._client, None, None
        if client is not None and hasattr(client, "aclose"):
            try:
                await client.aclose()
            except Exception:
                pass


def missing_tools(agent: str, available: list[str]) -> list[str]:
    """Allowlisted names that no connected server provides.

    Called at startup so a typo in ALLOWLISTS, or a server that is down, is reported once
    rather than showing up as an agent that mysteriously never calls a tool.
    """
    allowed = ALLOWLISTS.get(agent, ())
    if allowed == ("*",):
        return []
    return [n for n in allowed if n not in set(available)]
