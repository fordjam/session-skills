"""Writers for *synthetic* session files in each source's on-disk format.

Used by the test-suite and by ``session-skills demo``. All content here is fictional.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

Turn = tuple[str, str]  # (role, text) with role "user" or "assistant"


@dataclass
class Conversation:
    session_id: str
    day: date
    turns: Sequence[Turn]
    noise: bool = True  # add tool calls, reminders and other non-human records
    extra: list[dict[str, object]] = field(default_factory=list)  # raw extra records (format-specific)


def _stamp(day: date, minute: int) -> str:
    moment = datetime(day.year, day.month, day.day, 9, 0, tzinfo=timezone.utc) + timedelta(minutes=minute)
    return moment.isoformat().replace("+00:00", "Z")


def _ms(day: date, minute: int) -> int:
    moment = datetime(day.year, day.month, day.day, 9, 0, tzinfo=timezone.utc) + timedelta(minutes=minute)
    return int(moment.timestamp() * 1000)


# -- Claude Code -----------------------------------------------------------------------------

def claude_code_records(conv: Conversation) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    base = {"sessionId": conv.session_id, "cwd": "/work/demo", "isSidechain": False}
    for i, (role, text) in enumerate(conv.turns):
        stamp = _stamp(conv.day, i)
        if role == "user":
            content: object = text
            if conv.noise and i == 0:
                content = [
                    {"type": "text", "text": text},
                    {"type": "text", "text": "<system-reminder>Injected context the user never typed.</system-reminder>"},
                ]
            records.append({**base, "type": "user", "timestamp": stamp, "message": {"role": "user", "content": content}})
        else:
            blocks: list[dict[str, object]] = [{"type": "text", "text": text}]
            if conv.noise:
                blocks.append({"type": "tool_use", "id": f"tu{i}", "name": "Bash", "input": {"command": "make test"}})
            records.append({**base, "type": "assistant", "timestamp": stamp, "message": {"role": "assistant", "content": blocks}})
            if conv.noise:
                records.append(
                    {
                        **base,
                        "type": "user",
                        "timestamp": stamp,
                        "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"tu{i}", "content": "ok"}]},
                        "toolUseResult": {"stdout": "ok"},
                    }
                )
    if conv.noise:
        records.append(
            {**base, "type": "user", "isSidechain": True, "timestamp": _stamp(conv.day, 99),
             "message": {"role": "user", "content": "Sub-agent prompt written by the main agent."}}
        )
    return records + list(conv.extra)


def write_claude_code(root: Path, conv: Conversation, project: str = "-work-demo") -> Path:
    path = root / project / f"{conv.session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in claude_code_records(conv)), encoding="utf-8")
    return path


# -- Codex CLI ------------------------------------------------------------------------------

def codex_records(conv: Conversation, subagent: bool = False) -> list[dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "timestamp": _stamp(conv.day, 0),
            "type": "session_meta",
            "payload": {
                "id": conv.session_id,
                "timestamp": _stamp(conv.day, 0),
                "cwd": "/work/demo",
                "source": {"subagent": "review"} if subagent else "cli",
            },
        }
    ]
    if conv.noise:
        records.append(
            {
                "timestamp": _stamp(conv.day, 0),
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "<environment_context>cwd: /work/demo</environment_context>"}]},
            }
        )
    for i, (role, text) in enumerate(conv.turns):
        stamp = _stamp(conv.day, i)
        if role == "user":
            records.append({"timestamp": stamp, "type": "response_item", "payload": {
                "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})
            records.append({"timestamp": stamp, "type": "event_msg", "payload": {"type": "user_message", "message": text}})
        else:
            if conv.noise:
                records.append({"timestamp": stamp, "type": "response_item", "payload": {
                    "type": "function_call", "name": "shell", "arguments": "{\"command\":[\"ls\"]}", "call_id": f"c{i}"}})
                records.append({"timestamp": stamp, "type": "response_item", "payload": {
                    "type": "function_call_output", "call_id": f"c{i}", "output": "README.md"}})
            records.append({"timestamp": stamp, "type": "response_item", "payload": {
                "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}})
    return records + list(conv.extra)


def write_codex(root: Path, conv: Conversation, subagent: bool = False) -> Path:
    d = conv.day
    path = root / "sessions" / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}" / f"rollout-{conv.session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in codex_records(conv, subagent)), encoding="utf-8")
    return path


# -- OpenCode (SQLite) -----------------------------------------------------------------------

OPENCODE_SCHEMA = """
CREATE TABLE IF NOT EXISTS project (id text PRIMARY KEY, worktree text NOT NULL, time_created integer NOT NULL,
  time_updated integer NOT NULL, sandboxes text NOT NULL DEFAULT '[]');
CREATE TABLE IF NOT EXISTS session (id text PRIMARY KEY, project_id text NOT NULL, parent_id text, slug text NOT NULL,
  directory text NOT NULL, title text NOT NULL, version text NOT NULL, time_created integer NOT NULL,
  time_updated integer NOT NULL);
CREATE TABLE IF NOT EXISTS message (id text PRIMARY KEY, session_id text NOT NULL, time_created integer NOT NULL,
  time_updated integer NOT NULL, data text NOT NULL);
CREATE TABLE IF NOT EXISTS part (id text PRIMARY KEY, message_id text NOT NULL, session_id text NOT NULL,
  time_created integer NOT NULL, time_updated integer NOT NULL, data text NOT NULL);
"""


def write_opencode_db(path: Path, conversations: Sequence[Conversation], child_of: dict[str, str] | None = None) -> Path:
    """Write sessions into an OpenCode-style SQLite database. ``child_of`` maps a session id to its
    parent (a sub-agent session)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(OPENCODE_SCHEMA)
        conn.execute("INSERT OR IGNORE INTO project VALUES ('prj_demo', '/work/demo', 0, 0, '[]')")
        for conv in conversations:
            parent = (child_of or {}).get(conv.session_id)
            start = _ms(conv.day, 0)
            conn.execute(
                "INSERT INTO session VALUES (?, 'prj_demo', ?, ?, '/work/demo', 'demo', '1.0.0', ?, ?)",
                (conv.session_id, parent, conv.session_id, start, _ms(conv.day, len(conv.turns))),
            )
            for i, (role, text) in enumerate(conv.turns):
                mid = f"msg_{conv.session_id}_{i:03d}"
                t = _ms(conv.day, i)
                conn.execute("INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                             (mid, conv.session_id, t, t, json.dumps({"role": role, "time": {"created": t}})))
                parts: list[dict[str, object]] = [{"type": "text", "text": text}]
                if conv.noise and role == "user":
                    parts.append({"type": "text", "text": "Contents of a file the harness attached.", "synthetic": True})
                if conv.noise and role == "assistant":
                    parts.insert(0, {"type": "reasoning", "text": "thinking about it"})
                    parts.append({"type": "tool", "tool": "bash", "state": {"status": "completed", "output": "ok"}})
                for j, part in enumerate(parts):
                    conn.execute("INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                                 (f"prt_{mid}_{j}", mid, conv.session_id, t, t, json.dumps(part)))
        conn.commit()
    finally:
        conn.close()
    return path


def write_opencode_storage(root: Path, conv: Conversation, parent: str | None = None) -> Path:
    """Legacy (pre-SQLite) JSON layout under ``<data>/storage``."""
    info: dict[str, object] = {"id": conv.session_id, "time": {"created": _ms(conv.day, 0), "updated": _ms(conv.day, 9)}}
    if parent:
        info["parentID"] = parent
    session_file = root / "session" / "prj_demo" / f"{conv.session_id}.json"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(json.dumps(info), encoding="utf-8")
    for i, (role, text) in enumerate(conv.turns):
        mid = f"msg_{conv.session_id}_{i:03d}"
        mfile = root / "message" / conv.session_id / f"{mid}.json"
        mfile.parent.mkdir(parents=True, exist_ok=True)
        mfile.write_text(json.dumps({"id": mid, "sessionID": conv.session_id, "role": role,
                                     "time": {"created": _ms(conv.day, i)}}), encoding="utf-8")
        pfile = root / "part" / mid / f"prt_{mid}_0.json"
        pfile.parent.mkdir(parents=True, exist_ok=True)
        pfile.write_text(json.dumps({"id": f"prt_{mid}_0", "messageID": mid, "sessionID": conv.session_id,
                                     "type": "text", "text": text}), encoding="utf-8")
    return session_file


# -- demo corpus -----------------------------------------------------------------------------

_DEMO: list[tuple[int, list[Turn]]] = [
    (2, [("user", "add a --dry-run flag to the export command"), ("assistant", "Here is a detailed plan in nine sections..."),
         ("user", "way too long, just show me the diff"), ("assistant", "Diff below."), ("user", "run the tests")]),
    (4, [("user", "why does the nightly build fail on the parser step"), ("assistant", "Use the --legacy-parse flag."),
         ("user", "that flag doesn't exist in this version, check the help output"), ("assistant", "You're right, checking."),
         ("user", "status?")]),
    (6, [("user", "fix the flaky date test"), ("assistant", "Fixed! Everything passes now."),
         ("user", "did you run the tests? it still fails on CI"), ("assistant", "I had not run them. Running now."),
         ("user", "run the tests")]),
    (9, [("user", "refactor the loader, and I said no new dependencies"), ("assistant", "Added a YAML library to simplify it."),
         ("user", "I said no new dependencies. remove it"), ("assistant", "Removed."), ("user", "commit and push")]),
    (12, [("user", "summarise what changed in the cache module"), ("assistant", "A very long walkthrough of every function..."),
          ("user", "too long. give me the short version"), ("assistant", "Three changes: ..."), ("user", "status?")]),
    (15, [("user", "the export is wrong, the totals are doubled"), ("assistant", "Fixed the totals."),
          ("user", "still wrong, it doesn't work for empty files"), ("assistant", "Handled empty files."), ("user", "run the tests")]),
    (18, [("user", "use the retry helper from utils"), ("assistant", "I used retry_forever from the http client."),
          ("user", "that function doesn't exist. check the code before naming things"), ("assistant", "Correct, it is retry() in utils."),
          ("user", "commit and push")]),
    (21, [("user", "update the changelog"), ("assistant", "Done, all tests pass."),
          ("user", "did you test that? CI is red"), ("assistant", "I did not run them; running now."), ("user", "status?")]),
    (25, [("user", "rename the config key, I told you to keep the old one as an alias"), ("assistant", "Renamed and removed the old key."),
          ("user", "not what I asked, keep the alias"), ("assistant", "Restored the alias."), ("user", "commit and push")]),
    (28, [("user", "explain the migration plan"), ("assistant", "An extremely verbose essay..."),
          ("user", "too verbose, bullet points please"), ("assistant", "- step one\n- step two"), ("user", "perfect, thanks")]),
    (33, [("user", "the parser is wrong for nested quotes"), ("assistant", "Rewrote the tokenizer."),
          ("user", "still wrong, and the fix doesn't work on windows paths"), ("assistant", "Adjusted."), ("user", "run the tests")]),
    (36, [("user", "I said to keep the public API unchanged, you renamed two functions"), ("assistant", "Reverted the renames."),
          ("user", "run the tests")]),
    (38, [("user", "set the cache size in the config"), ("assistant", "Set cache.max_items in settings.toml."),
          ("user", "there is no such option in the schema, it doesn't exist"), ("assistant", "Correct; the option is cache_size.")]),
    (40, [("user", "did you run the linter before saying it was clean?"), ("assistant", "No, running it now."),
          ("user", "status?")]),
    (90, [("user", "too much jargon, plain english please"), ("assistant", "Simpler version.")]),
    (95, [("user", "what does that mean? use plain language"), ("assistant", "Put simply...")]),
    (99, [("user", "less jargon, explain it in plain english"), ("assistant", "Sure.")]),
]


def write_demo_sources(root: Path, today: date) -> dict[str, list[str]]:
    """Write a fictional corpus spread over three sources; return config paths per source."""
    claude_root, codex_root, opencode_db = root / "claude" / "projects", root / "codex", root / "opencode" / "opencode.db"
    db_convs: list[Conversation] = []
    for index, (days_ago, turns) in enumerate(_DEMO):
        conv = Conversation(f"demo{index:02d}", today - timedelta(days=days_ago), turns)
        bucket = index % 3
        if bucket == 0:
            write_claude_code(claude_root, conv)
        elif bucket == 1:
            write_codex(codex_root, conv)
        else:
            db_convs.append(conv)
    write_opencode_db(opencode_db, db_convs)
    return {
        "claude_code": [str(claude_root / "**" / "*.jsonl")],
        "codex": [str(codex_root / "sessions" / "**" / "*.jsonl")],
        "opencode": [str(opencode_db)],
    }
