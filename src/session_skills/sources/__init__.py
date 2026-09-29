"""Session-source adapters. Each adapter auto-discovers its default location; paths are configurable."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from .base import TOOL_MARKER, Adapter, Message, Session
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


__all__ = ["ADAPTERS", "TOOL_MARKER", "Adapter", "Message", "Session", "make_adapter"]
