"""Claude Code transcripts: ``~/.claude/projects/<project>/<session-id>.jsonl``.

One JSON object per line. Relevant fields: ``type`` ("user", "assistant", "summary", "system",
...), ``message.role``, ``message.content`` (string or list of blocks), ``isSidechain`` (sub-agent
turn), ``isMeta`` (injected, e.g. slash-command expansions), ``isCompactSummary``,
``toolUseResult`` and ``timestamp``. Files under a ``subagents/`` directory are sub-agent
transcripts and are skipped.
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from .base import Message, Session, clean_human_text, clip_assistant, finish, sha256_bytes

NAME = "claude_code"


def default_paths() -> list[str]:
    root = os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude"
    return [str(Path(root) / "projects" / "**" / "*.jsonl")]


def _text_blocks(content: Any) -> tuple[str, bool]:
    """Return (text, had_tool_blocks)."""
    if isinstance(content, str):
        return content, False
    if not isinstance(content, list):
        return "", False
    parts: list[str] = []
    tool = False
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif kind in {"tool_use", "tool_result", "server_tool_use", "web_search_tool_result"}:
                tool = True
    return "\n".join(parts), tool


def parse_lines(lines: Sequence[str]) -> list[Message]:
    messages: list[Message] = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("type") not in {"user", "assistant"}:
            continue
        if event.get("isSidechain") or event.get("isMeta") or event.get("isCompactSummary"):
            continue
        if event.get("isVisibleInTranscriptOnly") or "toolUseResult" in event:
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role") or event.get("type")
        text, had_tool = _text_blocks(message.get("content"))
        stamp = str(event.get("timestamp") or "")
        if role == "user":
            if had_tool:
                continue  # tool results travel in user turns
            cleaned = clean_human_text(text)
            if cleaned:
                messages.append(Message("user", cleaned, stamp))
        elif role == "assistant":
            clipped = clip_assistant(text)
            if clipped:
                messages.append(Message("assistant", clipped, stamp))
    return messages


class ClaudeCodeAdapter:
    name = NAME

    def __init__(self, paths: Sequence[str] | None = None):
        self.paths = list(paths) if paths else default_paths()
        self.warnings: list[str] = []

    def files(self) -> list[Path]:
        found: set[Path] = set()
        for pattern in self.paths:
            for hit in glob.glob(os.path.expanduser(pattern), recursive=True):
                path = Path(hit)
                if (
                    path.is_file()
                    and path.suffix == ".jsonl"
                    and "subagents" not in path.parts
                    and not path.name.startswith("agent-")
                ):
                    found.add(path)
        return sorted(found)

    def iter_sessions(self) -> Iterator[Session]:
        for path in self.files():
            try:
                raw = path.read_bytes()
            except OSError as exc:
                self.warnings.append(f"claude_code: could not read {path.name}: {exc.strerror or exc}")
                continue
            lines = raw.decode("utf-8", errors="replace").splitlines()
            # the file name is the session id: stable across moves, unlike the path
            key = path.stem
            session = Session(NAME, key, sha256_bytes(raw), parse_lines(lines))
            done = finish(session)
            if done:
                yield done
