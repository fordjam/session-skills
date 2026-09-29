"""Session-source adapters. Each adapter auto-discovers its default location; paths are configurable."""

from __future__ import annotations

from typing import Sequence, cast

from .base import Adapter, Message, Session, TOOL_MARKER
from .claude_code import ClaudeCodeAdapter
from .codex import CodexAdapter
from .opencode import OpenCodeAdapter

ADAPTERS = {
    "claude_code": ClaudeCodeAdapter,
    "codex": CodexAdapter,
    "opencode": OpenCodeAdapter,
}


def make_adapter(name: str, paths: Sequence[str] | None = None) -> Adapter:
    try:
        cls = ADAPTERS[name]
    except KeyError:
        raise ValueError(f"unknown source {name!r}; choose from {', '.join(ADAPTERS)}") from None
    return cast(Adapter, cls(paths))


__all__ = ["ADAPTERS", "Adapter", "Message", "Session", "TOOL_MARKER", "make_adapter"]
