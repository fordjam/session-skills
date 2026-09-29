"""Private, resumable, transactional state.

Everything lives in one SQLite database, ``state.db``, inside a 0700 directory that must be
outside any repository. The database file is 0600 and uses ``secure_delete`` so text removed by a
later redaction pass is overwritten on disk rather than left in free pages.

Tables:
  chats         one row per chat: redacted messages, dates, tracker status (new / partial /
                tagged / failed), flag and note
  tags          per-chat answers from the tagging backend, with provenance
                (missing = not tagged, never 0)
  part_answers  durable per-part answers, so a multi-part chat never pays twice for a part
  labels        second-backend labels for the calibration sample, with provenance
  docs          small JSON documents (calibration report, repeated asks, redaction policy)

Every update that must be atomic (a chat's status together with its tags) is one SQLite
transaction. A run holds an exclusive lock on ``<state>/.lock`` so two runs cannot interleave.
``tracker.csv`` and ``out/`` are derived views, written with unique temporary names and renamed
into place.
"""

from __future__ import annotations

import contextlib
import csv
import errno
import fcntl
import hashlib
import io
import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "2"

TRACKER_FIELDS = (
    "chat_id",
    "source",
    "file_hash",
    "date",
    "last_date",
    "human_messages",
    "chars",
    "parts",
    "status",
    "flagged",
    "note",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chats (
  chat_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  file_hash TEXT NOT NULL,
  date TEXT NOT NULL DEFAULT '',
  last_date TEXT NOT NULL DEFAULT '',
  human_messages INTEGER NOT NULL DEFAULT 0,
  chars INTEGER NOT NULL DEFAULT 0,
  parts INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'new',
  flagged TEXT NOT NULL DEFAULT '',
  note TEXT NOT NULL DEFAULT '',
  messages TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS tags (chat_id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS labels (chat_id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS part_answers (
  chat_id TEXT NOT NULL,
  role TEXT NOT NULL,
  cache_key TEXT NOT NULL,
  part INTEGER NOT NULL,
  answer TEXT NOT NULL,
  PRIMARY KEY (chat_id, role, cache_key, part)
);
CREATE TABLE IF NOT EXISTS docs (name TEXT PRIMARY KEY, data TEXT NOT NULL);
"""

_CHAT_COLUMNS = TRACKER_FIELDS


class StateError(RuntimeError):
    pass


def inside_repo(path: Path) -> Path | None:
    """Return the enclosing git/hg/svn work tree root, if any."""
    current = path.resolve()
    for candidate in (current, *current.parents):
        for marker in (".git", ".hg", ".svn"):
            if (candidate / marker).exists():
                return candidate
    return None


def chat_id(source: str, key: str) -> str:
    return hashlib.sha256(f"{source}\0{key}".encode()).hexdigest()[:16]


class State:
    def __init__(self, root: Path, *, create: bool = True):
        self.root = root.expanduser()
        repo = inside_repo(self.root)
        if repo is not None:
            raise StateError(
                f"refusing to use a state directory inside a repository ({repo}); "
                "choose a path outside any repo with --state-dir or SESSION_SKILLS_STATE"
            )
        if create:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.root, 0o700)
        self._conn: sqlite3.Connection | None = None
        self._lock_fd: int | None = None
        # Test hook: runs inside the commit transaction, after the status update and before the
        # tag write, so tests can simulate a crash at the worst possible moment.
        self.mid_commit_hook: Callable[[], None] | None = None

    # -- database ----------------------------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.root / "state.db"

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self.db_path.exists():
                fd = os.open(self.db_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(fd)
            os.chmod(self.db_path, 0o600)
            conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
            conn.execute("PRAGMA secure_delete = ON")
            conn.executescript(_SCHEMA)
            conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('schema', ?)", (SCHEMA_VERSION,))
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self.unlock()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One atomic unit of work: everything inside commits together or not at all."""
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    # -- run lock ----------------------------------------------------------------------------
    def lock(self) -> None:
        """Take the exclusive run lock, or fail fast if another run holds it."""
        if self._lock_fd is not None:
            return
        fd = os.open(self.root / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                raise StateError(f"another session-skills run is using {self.root}; wait for it to finish") from None
            raise
        self._lock_fd = fd

    def unlock(self) -> None:
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    # -- paths / files -------------------------------------------------------------------------
    def path(self, name: str) -> Path:
        return self.root / name

    @property
    def out(self) -> Path:
        out = self.root / "out"
        out.mkdir(exist_ok=True, mode=0o700)
        os.chmod(out, 0o700)
        return out

    def _private_dirs(self, directory: Path) -> None:
        """Create ``directory`` (and parents below the state root) as 0700."""
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        current = directory
        while current != self.root and self.root in current.parents:
            os.chmod(current, 0o700)
            current = current.parent

    def write_text(self, name: str | Path, text: str) -> Path:
        """Write a derived file atomically: unique temp file in the same directory, then rename."""
        target = name if isinstance(name, Path) else self.path(name)
        self._private_dirs(target.parent)
        fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return target

    # -- small JSON documents ------------------------------------------------------------------
    def read_json(self, name: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT data FROM docs WHERE name = ?", (name,)).fetchone()
        if not row:
            return {}
        try:
            value = json.loads(row[0])
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}

    def write_json(self, name: str, value: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO docs (name, data) VALUES (?, ?)",
                (name, json.dumps(value, sort_keys=True, ensure_ascii=False)),
            )

    # -- chats and tracker -----------------------------------------------------------------------
    def chats(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute("SELECT chat_id, source, date, last_date, messages FROM chats ORDER BY chat_id")
        return {
            cid: {"chat_id": cid, "source": source, "date": date, "last_date": last, "messages": json.loads(messages)}
            for cid, source, date, last, messages in rows
        }

    def tracker(self) -> dict[str, dict[str, str]]:
        rows = self.conn.execute(f"SELECT {', '.join(_CHAT_COLUMNS)} FROM chats ORDER BY chat_id")
        return {row[0]: {k: str(v) for k, v in zip(_CHAT_COLUMNS, row, strict=True)} for row in rows}

    def save_corpus(
        self,
        tracker: dict[str, dict[str, str]],
        chats: dict[str, dict[str, Any]],
        drop_tags: Iterable[str] = (),
    ) -> None:
        """Upsert chats and drop tags of changed chats in one transaction (``build-list``)."""
        placeholders = ", ".join("?" * (len(_CHAT_COLUMNS) + 1))
        with self.transaction() as conn:
            for cid in sorted(tracker):
                row = tracker[cid]
                conn.execute(
                    f"INSERT OR REPLACE INTO chats ({', '.join(_CHAT_COLUMNS)}, messages) VALUES ({placeholders})",
                    (
                        cid,
                        row.get("source", ""),
                        row.get("file_hash", ""),
                        row.get("date", ""),
                        row.get("last_date", ""),
                        int(row.get("human_messages") or 0),
                        int(row.get("chars") or 0),
                        int(row.get("parts") or 1),
                        row.get("status") or "new",
                        row.get("flagged", ""),
                        row.get("note", ""),
                        json.dumps(chats[cid].get("messages", []), ensure_ascii=False),
                    ),
                )
            for cid in drop_tags:
                conn.execute("DELETE FROM tags WHERE chat_id = ?", (cid,))
                conn.execute("DELETE FROM part_answers WHERE chat_id = ?", (cid,))

    def update_messages(self, updates: dict[str, list[dict[str, Any]]]) -> None:
        """Rewrite stored messages (a new redaction policy) in one transaction."""
        with self.transaction() as conn:
            for cid, messages in updates.items():
                conn.execute(
                    "UPDATE chats SET messages = ? WHERE chat_id = ?",
                    (json.dumps(messages, ensure_ascii=False), cid),
                )

    def rewrite_docs(self, transform: Callable[[Any], Any], keep: Iterable[str] = ()) -> int:
        """Apply ``transform`` to every stored JSON document except ``keep``; return how many
        changed. Used to re-redact derived documents when the redaction policy changes."""
        skip = set(keep)
        changed = 0
        with self.transaction() as conn:
            for name, data in conn.execute("SELECT name, data FROM docs").fetchall():
                if name in skip:
                    continue
                value = json.loads(data)
                new = transform(value)
                if new != value:
                    changed += 1
                    conn.execute(
                        "UPDATE docs SET data = ? WHERE name = ?",
                        (json.dumps(new, sort_keys=True, ensure_ascii=False), name),
                    )
        return changed

    def vacuum(self) -> None:
        """Rewrite the database file so no free page keeps text that was replaced."""
        self.conn.execute("VACUUM")

    def set_status(self, cid: str, status: str, flagged: str = "", note: str = "") -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE chats SET status = ?, flagged = ?, note = ? WHERE chat_id = ?",
                (status, flagged, note, cid),
            )

    def commit_tag(self, cid: str, flagged: str, note: str, tag: dict[str, Any]) -> None:
        """Mark a chat tagged and store its tag atomically; drop its per-part scratch answers."""
        with self.transaction() as conn:
            conn.execute(
                "UPDATE chats SET status = 'tagged', flagged = ?, note = ? WHERE chat_id = ?",
                (flagged, note, cid),
            )
            if self.mid_commit_hook is not None:
                self.mid_commit_hook()
            conn.execute(
                "INSERT OR REPLACE INTO tags (chat_id, data) VALUES (?, ?)",
                (cid, json.dumps(tag, sort_keys=True, ensure_ascii=False)),
            )
            conn.execute("DELETE FROM part_answers WHERE chat_id = ? AND role = 'tag'", (cid,))

    def export_tracker(self) -> Path:
        """Write ``tracker.csv``, a read-only view of the chat table."""
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=TRACKER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in self.tracker().values():
            writer.writerow({f: row.get(f, "") for f in TRACKER_FIELDS})
        return self.write_text("tracker.csv", buffer.getvalue())

    # -- tags, labels, part answers ------------------------------------------------------------
    def tags(self) -> dict[str, dict[str, Any]]:
        return {cid: json.loads(data) for cid, data in self.conn.execute("SELECT chat_id, data FROM tags")}

    def labels(self) -> dict[str, dict[str, Any]]:
        return {cid: json.loads(data) for cid, data in self.conn.execute("SELECT chat_id, data FROM labels")}

    def save_label(self, cid: str, label: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO labels (chat_id, data) VALUES (?, ?)",
                (cid, json.dumps(label, sort_keys=True, ensure_ascii=False)),
            )
            conn.execute("DELETE FROM part_answers WHERE chat_id = ? AND role = 'label'", (cid,))

    def part_answers(self, cid: str, role: str, cache_key: str) -> dict[int, dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT part, answer FROM part_answers WHERE chat_id = ? AND role = ? AND cache_key = ?",
            (cid, role, cache_key),
        )
        return {int(part): json.loads(answer) for part, answer in rows}

    def save_part_answer(self, cid: str, role: str, cache_key: str, part: int, answer: dict[str, Any]) -> None:
        """Durably store one part's answer; answers under an older cache key are dropped."""
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM part_answers WHERE chat_id = ? AND role = ? AND cache_key != ?",
                (cid, role, cache_key),
            )
            conn.execute(
                "INSERT OR REPLACE INTO part_answers (chat_id, role, cache_key, part, answer) VALUES (?, ?, ?, ?, ?)",
                (cid, role, cache_key, part, json.dumps(answer, sort_keys=True)),
            )
