"""``build-list``: discover sessions, redact them, and update the resumable tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .config import Config
from .redact import RedactOptions, redact
from .sources import Adapter, Session, make_adapter
from .state import State, chat_id


@dataclass
class BuildSummary:
    found: dict[str, int] = field(default_factory=dict)
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    total: int = 0
    chars: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "new": self.new,
            "changed": self.changed,
            "unchanged": self.unchanged,
            "total_chats": self.total,
            "estimated_input_tokens": self.chars // 4,
        }


def render(messages: Iterable[dict[str, Any]]) -> list[str]:
    """One line per message; the format every backend sees."""
    lines = []
    for message in messages:
        prefix = "USER" if message.get("role") == "user" else "ASSISTANT"
        text = " / ".join(part.strip() for part in str(message.get("text", "")).splitlines() if part.strip())
        lines.append(f"{prefix}: {text}")
    return lines


def split_parts(lines: Sequence[str], max_chars: int) -> list[str]:
    """Pack lines into chunks of at most ``max_chars`` characters (long lines are cut)."""
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        pieces = [line[i : i + max_chars] for i in range(0, len(line), max_chars)] or [""]
        for piece in pieces:
            if current and size + len(piece) + 1 > max_chars:
                parts.append("\n".join(current))
                current, size = [], 0
            current.append(piece)
            size += len(piece) + 1
    if current:
        parts.append("\n".join(current))
    return parts


def redact_session(session: Session, opts: RedactOptions) -> list[dict[str, Any]]:
    return [
        {"role": m.role, "text": redact(m.text, opts), "date": m.timestamp[:10] if m.timestamp else ""}
        for m in session.messages
    ]


def adapters_for(cfg: Config, only: Sequence[str] = ()) -> list[Adapter]:
    result = []
    for name, source in cfg.sources.items():
        if only and name not in only:
            continue
        if not source.enabled and not only:
            continue
        result.append(make_adapter(name, source.paths or None))
    return result


def build_list(cfg: Config, state: State, only: Sequence[str] = ()) -> BuildSummary:
    opts = cfg.redact_options()
    tracker = state.tracker()
    chats = state.chats()
    tags = state.tags()
    summary = BuildSummary()
    for adapter in adapters_for(cfg, only):
        count = 0
        for session in adapter.iter_sessions():
            count += 1
            cid = chat_id(session.source, session.key)
            messages = redact_session(session, opts)
            lines = render(messages)
            chars = sum(len(line) + 1 for line in lines)
            prior = tracker.get(cid)
            if prior and prior.get("file_hash") == session.fingerprint:
                summary.unchanged += 1
                status, flagged, note = prior.get("status", "new"), prior.get("flagged", ""), prior.get("note", "")
            else:
                if prior:
                    summary.changed += 1
                    tags.pop(cid, None)  # answers were for an older version of the chat
                else:
                    summary.new += 1
                status, flagged, note = "new", "", ""
            tracker[cid] = {
                "chat_id": cid,
                "source": session.source,
                "file_hash": session.fingerprint,
                "date": session.started,
                "human_messages": str(len(session.human_messages)),
                "chars": str(chars),
                "parts": str(len(split_parts(lines, cfg.max_chars))),
                "status": status,
                "flagged": flagged,
                "note": note,
            }
            chats[cid] = {
                "chat_id": cid,
                "source": session.source,
                "date": session.started,
                "messages": messages,
            }
        summary.found[adapter.name] = count
    summary.total = len(tracker)
    summary.chars = sum(int(r.get("chars") or 0) for r in tracker.values() if r.get("status") != "tagged")
    state.save_chats(chats)
    state.save_tracker(tracker)
    state.save_tags(tags)
    return summary
