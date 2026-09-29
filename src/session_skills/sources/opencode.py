"""OpenCode sessions.

ASSUMPTIONS (derived from OpenCode's source, not from any real session; override with config if
your install differs):

* Data dir: ``$XDG_DATA_HOME/opencode`` (default ``~/.local/share/opencode``).
* Current releases (1.x) keep sessions in SQLite: ``opencode.db`` (or ``opencode-<channel>.db``,
  or the file named by ``$OPENCODE_DB``). Tables used, read-only:

  - ``session(id, parent_id, time_created, time_updated, ...)`` -- ``parent_id`` set means a
    sub-agent (child) session, which is skipped;
  - ``message(id, session_id, time_created, data)`` -- ``data`` is JSON with ``role``;
  - ``part(id, message_id, session_id, time_created, data)`` -- ``data`` is JSON with ``type``
    (``text``, ``tool``, ``reasoning``, ``file``, ``step-start``, ...), ``text`` and an optional
    ``synthetic`` flag for text the harness injected.

* Older releases used JSON files under ``<data>/storage``: ``session/<project>/<id>.json``
  (``parentID`` marks children), ``message/<session>/<id>.json`` and
  ``part/<message>/<id>.json`` with the same fields in camelCase.

Only non-synthetic ``text`` parts are kept. OpenCode does not mark slash-command expansions, so
a custom command's template text is indistinguishable from typed text (known limitation).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from .base import Message, Session, clean_human_text, clip_assistant, finish, iso_date

NAME = "opencode"


def data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(base).expanduser() / "opencode"


def default_paths() -> list[str]:
    paths = []
    if os.environ.get("OPENCODE_DB"):
        paths.append(os.environ["OPENCODE_DB"])
    root = data_dir()
    paths += [str(root / "opencode*.db"), str(root / "storage")]
    return paths


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, bytes)):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _message_from_parts(role: Any, parts: list[dict[str, Any]], stamp: str) -> Message | None:
    texts = [
        str(p.get("text") or "")
        for p in parts
        if p.get("type") == "text" and not p.get("synthetic") and not p.get("ignored")
    ]
    text = "\n".join(t for t in texts if t)
    if role == "user":
        cleaned = clean_human_text(text)
        return Message("user", cleaned, stamp) if cleaned else None
    if role == "assistant":
        clipped = clip_assistant(text)
        return Message("assistant", clipped, stamp) if clipped else None
    return None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def read_sqlite(path: Path, warnings: list[str] | None = None) -> Iterator[Session]:
    notes = warnings if warnings is not None else []
    uri = f"file:{path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        notes.append(f"opencode: could not open {path.name}: {exc}")
        return
    try:
        cols = _columns(conn, "session")
        if not cols or not _columns(conn, "message") or not _columns(conn, "part"):
            notes.append(f"opencode: skipped {path.name}: no session/message/part tables (not an OpenCode database?)")
            return
        parent = "parent_id" if "parent_id" in cols else "NULL"
        updated = "time_updated" if "time_updated" in cols else "0"
        sessions = conn.execute(
            f"SELECT id, {parent}, time_created, {updated} FROM session ORDER BY time_created, id"
        ).fetchall()
        for session_id, parent_id, created, updated_at in sessions:
            if parent_id:
                continue
            rows = conn.execute(
                "SELECT id, time_created, data FROM message WHERE session_id = ? "
                "ORDER BY time_created, id",
                (session_id,),
            ).fetchall()
            digest = hashlib.sha256(f"{session_id}:{updated_at}".encode())
            messages: list[Message] = []
            for message_id, msg_created, data in rows:
                info = _json(data)
                parts = [
                    _json(p[0])
                    for p in conn.execute(
                        "SELECT data FROM part WHERE message_id = ? ORDER BY id", (message_id,)
                    )
                ]
                digest.update(f"{message_id}:{len(parts)}".encode())
                message = _message_from_parts(info.get("role"), parts, iso_date(msg_created))
                if message:
                    messages.append(message)
            session = Session(NAME, str(session_id), digest.hexdigest(), messages, iso_date(created))
            done = finish(session)
            if done:
                yield done
    except sqlite3.Error as exc:
        notes.append(f"opencode: error reading {path.name} (schema changed?): {exc}")
        return
    finally:
        conn.close()


def read_storage(root: Path) -> Iterator[Session]:
    for session_file in sorted(root.glob("session/*/*.json")):
        info = _json(_read(session_file))
        session_id = str(info.get("id") or session_file.stem)
        if info.get("parentID") or info.get("parent_id"):
            continue
        digest = hashlib.sha256(_read(session_file).encode())
        loaded: list[tuple[Any, str, dict[str, Any]]] = []
        for message_file in (root / "message" / session_id).glob("*.json"):
            text = _read(message_file)
            digest.update(text.encode())
            minfo = _json(text)
            created = (minfo.get("time") or {}).get("created", 0) if isinstance(minfo.get("time"), dict) else 0
            loaded.append((created, str(minfo.get("id") or message_file.stem), minfo))
        messages: list[Message] = []
        for created, message_id, minfo in sorted(loaded, key=lambda t: (t[0] or 0, t[1])):
            parts = []
            for part_file in sorted((root / "part" / message_id).glob("*.json")):
                text = _read(part_file)
                digest.update(text.encode())
                parts.append(_json(text))
            message = _message_from_parts(minfo.get("role"), parts, iso_date(created))
            if message:
                messages.append(message)
        time: dict[str, Any] = info["time"] if isinstance(info.get("time"), dict) else {}
        session = Session(NAME, session_id, digest.hexdigest(), messages, iso_date(time.get("created")))
        done = finish(session)
        if done:
            yield done


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


class OpenCodeAdapter:
    name = NAME

    def __init__(self, paths: Sequence[str] | None = None):
        self.paths = list(paths) if paths else default_paths()
        self.warnings: list[str] = []

    def targets(self) -> list[tuple[str, Path]]:
        found: dict[Path, str] = {}
        for pattern in self.paths:
            for hit in glob.glob(os.path.expanduser(pattern), recursive=True):
                path = Path(hit)
                if path.is_file() and path.suffix == ".db":
                    found[path] = "sqlite"
                elif path.is_dir() and (path / "session").is_dir():
                    found[path] = "storage"
        return sorted(((kind, path) for path, kind in found.items()), key=lambda t: str(t[1]))

    def iter_sessions(self) -> Iterator[Session]:
        for kind, path in self.targets():
            yield from read_sqlite(path, self.warnings) if kind == "sqlite" else read_storage(path)
