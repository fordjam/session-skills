"""``build-list``: discover sessions, redact them, and update the resumable tracker."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .redact import RedactOptions, policy_fingerprint, redact
from .sources import Adapter, Session, make_adapter
from .state import State, chat_id


@dataclass
class BuildSummary:
    found: dict[str, int] = field(default_factory=dict)
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    duplicates: int = 0
    total: int = 0
    chars: int = 0
    re_redacted: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "new": self.new,
            "changed": self.changed,
            "unchanged": self.unchanged,
            "duplicates": self.duplicates,
            "total_chats": self.total,
            "re_redacted": self.re_redacted,
            "estimated_input_tokens": self.chars // 4,
            "warnings": self.warnings,
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


def event_dates(messages: Sequence[dict[str, Any]], fallback: str) -> tuple[str, str]:
    """(first, last) message date; a chat's recency is its latest message, not its start."""
    dates = sorted(str(m.get("date") or "") for m in messages if m.get("date"))
    return (dates[0] if dates else fallback, dates[-1] if dates else fallback)


def _redact_json(value: Any, opts: RedactOptions) -> Any:
    if isinstance(value, str):
        return redact(value, opts)
    if isinstance(value, list):
        return [_redact_json(v, opts) for v in value]
    if isinstance(value, dict):
        return {k: _redact_json(v, opts) for k, v in value.items()}
    return value


def refresh_redaction(cfg: Config, state: State) -> int:
    """Apply a changed redaction policy (for example a new deny word) to everything already
    stored: chat text (including chats whose source files are gone), derived documents in the
    database such as the repeated-ask report, and generated files under ``out/``. The database
    is then compacted so no free page keeps the old text. Returns how many chats changed.

    Not touched: ``backups/``, which holds copies of your own instruction files taken by
    ``install``."""
    opts = cfg.redact_options()
    fingerprint = policy_fingerprint(opts)
    if state.read_json("redaction_policy").get("fingerprint") == fingerprint:
        return 0
    updates: dict[str, list[dict[str, Any]]] = {}
    for cid, chat in state.chats().items():
        messages = chat.get("messages", [])
        cleaned = [{**m, "text": redact(str(m.get("text", "")), opts)} for m in messages]
        if cleaned != messages:
            updates[cid] = cleaned
    if updates:
        state.update_messages(updates)
    state.rewrite_docs(lambda value: _redact_json(value, opts), keep=("redaction_policy",))
    out = state.root / "out"
    if out.is_dir():
        for path in sorted(out.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                text = path.read_text(encoding="utf-8", errors="replace")
                cleaned_text = _redact_lines(text, opts)
                if cleaned_text != text:
                    state.write_text(path, cleaned_text)
    state.write_json("redaction_policy", {"fingerprint": fingerprint})
    state.vacuum()
    return len(updates)


def _redact_lines(text: str, opts: RedactOptions) -> str:
    """Redact a generated file line by line, so the pasted-block rule never collapses ordinary
    report tables."""
    return "\n".join(redact(line, opts) if line.strip() else line for line in text.split("\n"))


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
    state.lock()
    summary = BuildSummary()
    summary.re_redacted = refresh_redaction(cfg, state)
    opts = cfg.redact_options()
    tracker = state.tracker()
    # One chat per session id. The same session found twice (a rollout copied into
    # archived_sessions, say) keeps the longer copy.
    sessions: dict[str, Session] = {}
    for adapter in adapters_for(cfg, only):
        count = 0
        for session in adapter.iter_sessions():
            count += 1
            cid = chat_id(session.source, session.key)
            prior = sessions.get(cid)
            if prior is not None:
                summary.duplicates += 1
                if len(prior.messages) >= len(session.messages):
                    continue
            sessions[cid] = session
        summary.found[adapter.name] = count
        summary.warnings.extend(getattr(adapter, "warnings", []))
    rows: dict[str, dict[str, str]] = {}
    chats: dict[str, dict[str, Any]] = {}
    changed: list[str] = []
    for cid, session in sorted(sessions.items()):
        messages = redact_session(session, opts)
        lines = render(messages)
        first, last = event_dates(messages, session.started)
        prior_row = tracker.get(cid)
        if prior_row and prior_row.get("file_hash") == session.fingerprint:
            summary.unchanged += 1
            status = prior_row.get("status", "new")
            flagged, note = prior_row.get("flagged", ""), prior_row.get("note", "")
        else:
            if prior_row:
                summary.changed += 1
                changed.append(cid)  # answers were for an older version of the chat
            else:
                summary.new += 1
            status, flagged, note = "new", "", ""
        rows[cid] = {
            "chat_id": cid,
            "source": session.source,
            "file_hash": session.fingerprint,
            "date": first,
            "last_date": last,
            "human_messages": str(len(session.human_messages)),
            "chars": str(sum(len(line) + 1 for line in lines)),
            "parts": str(len(split_parts(lines, cfg.max_chars))),
            "status": status,
            "flagged": flagged,
            "note": note,
        }
        chats[cid] = {"messages": messages}
    state.save_corpus(rows, chats, drop_tags=changed)
    tracker = state.tracker()
    summary.total = len(tracker)
    summary.chars = sum(int(r.get("chars") or 0) for r in tracker.values() if r.get("status") != "tagged")
    state.export_tracker()
    return summary
