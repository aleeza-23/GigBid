"""Load versioned prompts from disk.

The assignment requires every prompt to live in `agent/prompts/` as a versioned file, with
none inline in node code, and CI checks for it. The reason is practical rather than
stylistic: a prompt is the main thing you tune, so it needs to be diffable on its own and
its version needs to appear in a trace. "The Writer got worse last Tuesday" is answerable
when the trace records `writer@v3`.

Each file starts with a small frontmatter block:

    ---
    version: 3
    node: writer
    model: large
    notes: dropped the three-paragraph rule, it was fighting the evidence count
    ---

The body is a template using `{placeholders}`. `render` fills them and fails loudly on a
missing one, because a prompt silently containing the literal text `{requirement}` is a
bug that produces plausible output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parent
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


class PromptError(RuntimeError):
    """A missing prompt file, or a template rendered without all its inputs."""


@dataclass
class Prompt:
    node: str
    version: int
    body: str
    model: str = "small"
    notes: str = ""

    @property
    def label(self) -> str:
        """What goes in the trace: `writer@v3`."""
        return f"{self.node}@v{self.version}"

    def placeholders(self) -> set[str]:
        return set(_PLACEHOLDER.findall(self.body))

    def render(self, **values: Any) -> str:
        """Fill the template. Raises when an input is missing."""
        missing = self.placeholders() - set(values)
        if missing:
            raise PromptError(
                f"{self.label} needs {sorted(missing)}, which were not supplied. "
                "Rendering with a placeholder left in produces plausible nonsense."
            )
        out = self.body
        for key, value in values.items():
            out = out.replace("{" + key + "}", "" if value is None else str(value))
        return out.strip()


@lru_cache(maxsize=64)
def load(node: str) -> Prompt:
    """Load one node's prompt. Cached; call `reload` after editing a file."""
    path = PROMPTS_DIR / f"{node}.md"
    if not path.exists():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.md") if p.stem != "README")
        raise PromptError(f"no prompt for node {node!r}. Available: {available}")

    text = path.read_text(encoding="utf-8")
    meta: dict[str, Any] = {}
    match = _FRONTMATTER.match(text)
    body = text
    if match:
        body = text[match.end() :]
        for line in match.group(1).split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip().strip("\"'")

    return Prompt(
        node=str(meta.get("node") or node),
        version=int(meta.get("version") or 1),
        body=body.strip(),
        model=str(meta.get("model") or "small"),
        notes=str(meta.get("notes") or ""),
    )


def reload() -> None:
    load.cache_clear()


def all_prompts() -> list[Prompt]:
    return [
        load(path.stem)
        for path in sorted(PROMPTS_DIR.glob("*.md"))
        if path.stem != "README"
    ]


def render(node: str, **values: Any) -> tuple[str, str]:
    """Render a node's prompt, returning (text, label) so the label reaches the trace."""
    prompt = load(node)
    return prompt.render(**values), prompt.label
