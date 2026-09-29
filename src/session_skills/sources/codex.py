"""Codex CLI rollouts: ``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`` (+ ``archived_sessions``).

Current format: each line is ``{"timestamp", "type", "payload"}`` where ``type`` is
``session_meta``, ``response_item``, ``event_msg`` or ``turn_context``. Human text appears twice:
as ``event_msg`` / ``payload.type == "user_message"`` (exactly what was typed) and as a
``response_item`` message with role ``user`` (which also carries injected AGENTS.md and
environment context). When ``user_message`` events exist they are used; otherwise the
``response_item`` user messages are cleaned. Older rollouts have no envelope: each line *is* the
item (``{"type": "message", "role": ...}``). Sub-agent rollouts (``session_meta.payload.source``
mentions a sub-agent) are skipped.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

from .base import Message, Session, clean_human_text, clip_assistant, finish, iso_date, sha256_bytes

NAME = "codex"


def default_paths() -> list[str]:
    root = os.environ.get("CODEX_HOME") or "~/.codex"
    return [
        str(Path(root) / "sessions" / "**" / "*.jsonl"),
        str(Path(root) / "archived_sessions" / "**" / "*.jsonl"),
    ]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in {"input_text", "output_text", "text"}:
            value = block.get("text")
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(parts)


def _is_subagent(meta: dict[str, Any]) -> bool:
    source = meta.get("source")
    if isinstance(source, dict):
        return any("subagent" in str(k).lower() for k in source)
    return isinstance(source, str) and "subagent" in source.lower()


def parse_lines(lines: Sequence[str]) -> tuple[list[Message], str, bool]:
    """Return (messages, started_date, is_subagent)."""
    typed: list[Message] = []
    item_users: list[Message] = []
    assistants: list[Message] = []
    order: list[tuple[str, int]] = []  # interleave in file order
    started = ""
    subagent = False
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        stamp = str(event.get("timestamp") or "")
        kind = event.get("type")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else None
        if kind == "session_meta" and payload is not None:
            started = started or iso_date(payload.get("timestamp") or stamp)
            subagent = subagent or _is_subagent(payload)
            continue
        if payload is None and "type" in event and "role" in event:
            payload, kind = event, "response_item"  # legacy rollout line
        elif payload is None and "instructions" in event and "id" in event:
            started = started or iso_date(event.get("timestamp"))
            continue
        if payload is None:
            continue
        ptype = payload.get("type")
        if kind == "event_msg" and ptype == "user_message":
            text = clean_human_text(str(payload.get("message") or ""))
            if text:
                typed.append(Message("user", text, stamp))
                order.append(("typed", len(typed) - 1))
        elif kind == "response_item" and ptype == "message":
            role = payload.get("role")
            text = _content_text(payload.get("content"))
            if role == "user":
                cleaned = clean_human_text(text)
                if cleaned:
                    item_users.append(Message("user", cleaned, stamp))
                    order.append(("item", len(item_users) - 1))
            elif role == "assistant":
                clipped = clip_assistant(text)
                if clipped:
                    assistants.append(Message("assistant", clipped, stamp))
                    order.append(("assistant", len(assistants) - 1))
        # function_call, function_call_output, reasoning, local_shell_call, ... are dropped
    use_typed = bool(typed)
    messages: list[Message] = []
    for bucket, index in order:
        if bucket == "assistant":
            messages.append(assistants[index])
        elif bucket == "typed" and use_typed:
            messages.append(typed[index])
        elif bucket == "item" and not use_typed:
            messages.append(item_users[index])
    return messages, started, subagent


class CodexAdapter:
    name = NAME

    def __init__(self, paths: Sequence[str] | None = None):
        self.paths = list(paths) if paths else default_paths()

    def files(self) -> list[Path]:
        found: set[Path] = set()
        for pattern in self.paths:
            for hit in glob.glob(os.path.expanduser(pattern), recursive=True):
                path = Path(hit)
                if path.is_file() and path.suffix == ".jsonl":
                    found.add(path)
        return sorted(found)

    def iter_sessions(self) -> Iterator[Session]:
        for path in self.files():
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            messages, started, subagent = parse_lines(raw.decode("utf-8", errors="replace").splitlines())
            if subagent:
                continue
            session = Session(NAME, str(path), sha256_bytes(raw), messages, started)
            done = finish(session)
            if done:
                yield done
