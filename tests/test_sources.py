from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from session_skills.sources import TOOL_MARKER, make_adapter
from session_skills.sources.base import ASSISTANT_CHARS, clean_human_text, iso_date
from session_skills.synthetic import (
    Conversation,
    write_claude_code,
    write_codex,
    write_opencode_db,
    write_opencode_storage,
)

DAY = date(2026, 9, 1)


def humans(session):
    return [m.text for m in session.messages if m.role == "user"]


def only(adapter):
    sessions = list(adapter.iter_sessions())
    assert len(sessions) == 1, sessions
    return sessions[0]


# -- Claude Code --------------------------------------------------------------------------

def test_claude_code_keeps_only_human_text(tmp_path: Path):
    root = tmp_path / "projects"
    long_reply = "A" * (ASSISTANT_CHARS + 50)
    conv = Conversation(
        "s1",
        DAY,
        [("user", "fix the parser"), ("assistant", long_reply), ("user", "too long, shorter please")],
        extra=[
            # slash command: only typed arguments survive; the expansion is isMeta
            {"type": "user", "timestamp": "2026-09-01T10:00:00Z", "message": {"role": "user", "content":
                "<command-message>review</command-message>\n<command-name>/review</command-name>\n<command-args>the auth module</command-args>"}},
            {"type": "user", "isMeta": True, "timestamp": "2026-09-01T10:00:01Z", "message": {"role": "user", "content": "EXPANDED COMMAND PROMPT"}},
            {"type": "user", "timestamp": "2026-09-01T10:00:02Z", "message": {"role": "user", "content": "<command-name>/clear</command-name>\n<command-args></command-args>"}},
            {"type": "user", "timestamp": "2026-09-01T10:00:03Z", "message": {"role": "user", "content": "<task-notification>background task finished</task-notification>"}},
            {"type": "user", "timestamp": "2026-09-01T10:00:04Z", "message": {"role": "user", "content": "[Request interrupted by user]"}},
            {"type": "user", "isCompactSummary": True, "timestamp": "2026-09-01T10:00:05Z", "message": {"role": "user", "content": "SUMMARY OF EARLIER"}},
            {"type": "summary", "summary": "title"},
            {"type": "user", "timestamp": "2026-09-01T10:00:06Z", "message": {"role": "user", "content": "<local-command-stdout>ok</local-command-stdout>"}},
            "not json",
        ],
    )
    path = write_claude_code(root, conv)
    with path.open("a") as handle:  # a malformed line must be ignored
        handle.write("{broken\n")
    # sub-agent transcripts in both layouts are skipped entirely
    write_claude_code(root / "-work-demo" / "s1" / "subagents", Conversation("agent-x", DAY, [("user", "SUBAGENT TEXT")]), project="")
    write_claude_code(root, Conversation("agent-abc", DAY, [("user", "OLD SUBAGENT TEXT")]))

    session = only(make_adapter("claude_code", [str(root / "**" / "*.jsonl")]))
    assert humans(session) == ["fix the parser", "too long, shorter please", "the auth module"]
    assistants = [m.text for m in session.messages if m.role == "assistant"]
    assert assistants == ["A" * ASSISTANT_CHARS]
    assert session.started == "2026-09-01"
    joined = json.dumps([m.text for m in session.messages])
    for injected in ("Injected context", "EXPANDED", "background task", "SUMMARY", "Sub-agent prompt", "SUBAGENT"):
        assert injected not in joined


def test_sessions_without_human_text_or_from_this_tool_are_skipped(tmp_path: Path):
    root = tmp_path / "projects"
    write_claude_code(root, Conversation("empty", DAY, [("assistant", "hello")]))
    write_claude_code(root, Conversation("ours", DAY, [("user", f"{TOOL_MARKER}\nlabel this"), ("assistant", "{}")]))
    assert list(make_adapter("claude_code", [str(root / "**" / "*.jsonl")]).iter_sessions()) == []


# -- Codex ----------------------------------------------------------------------------------

def test_codex_uses_typed_user_messages_and_drops_tools(tmp_path: Path):
    conv = Conversation("c1", DAY, [("user", "run the tests"), ("assistant", "All green."), ("user", "status?")])
    write_codex(tmp_path, conv)
    write_codex(tmp_path, Conversation("c2", DAY, [("user", "SUBAGENT ASK")]), subagent=True)
    session = only(make_adapter("codex", [str(tmp_path / "sessions" / "**" / "*.jsonl")]))
    assert humans(session) == ["run the tests", "status?"]
    assert [m.text for m in session.messages if m.role == "assistant"] == ["All green."]
    assert "environment_context" not in json.dumps([m.text for m in session.messages])
    assert session.started == "2026-09-01"


def test_codex_without_event_messages_cleans_response_items(tmp_path: Path):
    lines = [
        {"timestamp": "2026-09-02T09:00:00Z", "type": "session_meta", "payload": {"id": "x", "timestamp": "2026-09-02T09:00:00Z"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "# AGENTS.md instructions for /work\nbe nice"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "<user_instructions>x</user_instructions>"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "add logging"}]}},
        {"type": "response_item", "payload": {"type": "reasoning", "summary": []}},
    ]
    path = tmp_path / "sessions" / "rollout-x.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    session = only(make_adapter("codex", [str(tmp_path / "sessions" / "*.jsonl")]))
    assert humans(session) == ["add logging"]
    assert session.started == "2026-09-02"


def test_codex_legacy_rollout(tmp_path: Path):
    lines = [
        {"id": "legacy", "timestamp": "2025-05-01T10:00:00Z", "instructions": ""},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "explain this error"}]},
        {"type": "function_call", "name": "shell", "arguments": "{}"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "It means X."}]},
    ]
    path = tmp_path / "rollout-legacy.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    session = only(make_adapter("codex", [str(tmp_path / "*.jsonl")]))
    assert humans(session) == ["explain this error"]
    assert session.started == "2025-05-01"


# -- OpenCode -------------------------------------------------------------------------------

def test_opencode_sqlite(tmp_path: Path):
    db = tmp_path / "opencode.db"
    main = Conversation("ses_main", DAY, [("user", "rename the module"), ("assistant", "Renamed."), ("user", "run the tests")])
    child = Conversation("ses_child", DAY, [("user", "CHILD TASK TEXT")])
    write_opencode_db(db, [main, child], child_of={"ses_child": "ses_main"})
    session = only(make_adapter("opencode", [str(db)]))
    assert humans(session) == ["rename the module", "run the tests"]
    assert [m.text for m in session.messages if m.role == "assistant"] == ["Renamed."]
    text = json.dumps([m.text for m in session.messages])
    assert "harness attached" not in text and "thinking" not in text and "CHILD" not in text
    assert session.started == "2026-09-01"


def test_opencode_fingerprint_changes_when_session_grows(tmp_path: Path):
    db = tmp_path / "opencode.db"
    write_opencode_db(db, [Conversation("ses_a", DAY, [("user", "one")])])
    first = only(make_adapter("opencode", [str(db)])).fingerprint
    db.unlink()
    write_opencode_db(db, [Conversation("ses_a", DAY, [("user", "one"), ("assistant", "two"), ("user", "three")])])
    assert only(make_adapter("opencode", [str(db)])).fingerprint != first


def test_opencode_legacy_storage(tmp_path: Path):
    storage = tmp_path / "storage"
    write_opencode_storage(storage, Conversation("ses_old", DAY, [("user", "update the docs"), ("assistant", "Done.")]))
    write_opencode_storage(storage, Conversation("ses_kid", DAY, [("user", "CHILD")]), parent="ses_old")
    session = only(make_adapter("opencode", [str(storage)]))
    assert humans(session) == ["update the docs"]


def test_opencode_ignores_unrelated_sqlite(tmp_path: Path):
    import sqlite3

    db = tmp_path / "other.db"
    sqlite3.connect(db).execute("CREATE TABLE t (x)").connection.close()
    assert list(make_adapter("opencode", [str(db)]).iter_sessions()) == []


# -- helpers ----------------------------------------------------------------------------------

def test_clean_human_text_and_dates():
    assert clean_human_text("<system-reminder>x</system-reminder>  hi  ") == "hi"
    assert clean_human_text("<environment_context>x</environment_context>") == ""
    assert iso_date(1_788_000_000_000) == iso_date(1_788_000_000)
    assert iso_date("2026-09-01T23:00:00Z") == "2026-09-01"
    assert iso_date("garbage") == ""


def test_unreadable_opencode_database_is_reported_not_silently_dropped(tmp_path: Path, cfg, state, capsys):
    from session_skills import cli
    from session_skills.collect import build_list

    broken = tmp_path / "opencode" / "opencode.db"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"this is not a sqlite database at all" * 100)
    adapter = make_adapter("opencode", [str(broken)])
    assert list(adapter.iter_sessions()) == []
    assert adapter.warnings and "opencode.db" in adapter.warnings[0]

    summary = build_list(cfg, state)
    assert any("opencode.db" in w for w in summary.warnings)
    state.close()
    code = cli.run(["--state-dir", str(cfg.state_dir), "build-list", "--path", f"opencode={broken}", "--source", "opencode"])
    assert code == 1 and "warning: opencode" in capsys.readouterr().out


def test_unrelated_sqlite_is_noted_but_not_an_error(tmp_path: Path):
    import sqlite3

    db = tmp_path / "other.db"
    sqlite3.connect(db).execute("CREATE TABLE t (x)").connection.close()
    adapter = make_adapter("opencode", [str(db)])
    assert list(adapter.iter_sessions()) == []
    assert adapter.warnings and ": skipped " in adapter.warnings[0]


def test_session_keys_are_native_ids_not_paths(tmp_path: Path):
    write_codex(tmp_path, Conversation("rollout-id-1", DAY, [("user", "hello")]))
    session = only(make_adapter("codex", [str(tmp_path / "sessions" / "**" / "*.jsonl")]))
    assert session.key == "rollout-id-1"
    root = tmp_path / "projects"
    write_claude_code(root, Conversation("0000-uuid", DAY, [("user", "hello")]))
    assert only(make_adapter("claude_code", [str(root / "**" / "*.jsonl")])).key == "0000-uuid"
    db = tmp_path / "opencode.db"
    write_opencode_db(db, [Conversation("ses_x", DAY, [("user", "hello")])])
    assert only(make_adapter("opencode", [str(db)])).key == "ses_x"
