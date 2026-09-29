"""Private, resumable state: a 0700 directory outside any repository, files written 0600.

Files:
  chats.jsonl         redacted messages per chat (never raw text)
  tracker.csv         one row per chat: status new/tagged/failed, flagged, note
  tags.jsonl          per-chat answers from the tagging backend (missing = not tagged, never 0)
  labels.jsonl        second-backend labels for the calibration sample
  calibration.json    per-question agreement and threshold
  repeated.json       repeated-ask clusters
  out/                report.md, skills/, AGENTS.md block, CLAUDE.md snippet
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

TRACKER_FIELDS = (
    "chat_id",
    "source",
    "file_hash",
    "date",
    "human_messages",
    "chars",
    "parts",
    "status",
    "flagged",
    "note",
)


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

    # -- paths -----------------------------------------------------------------------------
    def path(self, name: str) -> Path:
        return self.root / name

    @property
    def out(self) -> Path:
        out = self.root / "out"
        out.mkdir(exist_ok=True, mode=0o700)
        os.chmod(out, 0o700)
        return out

    # -- low-level io ----------------------------------------------------------------------
    def _private_dirs(self, directory: Path) -> None:
        """Create ``directory`` (and parents below the state root) as 0700."""
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        current = directory
        while current != self.root and self.root in current.parents:
            os.chmod(current, 0o700)
            current = current.parent

    def write_text(self, name: str | Path, text: str) -> Path:
        target = name if isinstance(name, Path) else self.path(name)
        self._private_dirs(target.parent)
        tmp = target.with_name(target.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, target)
        os.chmod(target, 0o600)
        return target

    def read_jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self.path(name)
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def write_jsonl(self, name: str, rows: Iterable[dict[str, Any]]) -> None:
        self.write_text(name, "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))

    def read_json(self, name: str) -> dict[str, Any]:
        path = self.path(name)
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}

    def write_json(self, name: str, value: dict[str, Any]) -> None:
        self.write_text(name, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    # -- domain files ----------------------------------------------------------------------
    def chats(self) -> dict[str, dict[str, Any]]:
        return {r["chat_id"]: r for r in self.read_jsonl("chats.jsonl") if "chat_id" in r}

    def save_chats(self, chats: dict[str, dict[str, Any]]) -> None:
        self.write_jsonl("chats.jsonl", (chats[k] for k in sorted(chats)))

    def tracker(self) -> dict[str, dict[str, str]]:
        path = self.path("tracker.csv")
        if not path.is_file():
            return {}
        with path.open(newline="", encoding="utf-8") as handle:
            return {r["chat_id"]: dict(r) for r in csv.DictReader(handle) if r.get("chat_id")}

    def save_tracker(self, rows: dict[str, dict[str, str]]) -> None:
        import io

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=TRACKER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({f: rows[key].get(f, "") for f in TRACKER_FIELDS})
        self.write_text("tracker.csv", buffer.getvalue())

    def tags(self) -> dict[str, dict[str, Any]]:
        return {r["chat_id"]: r for r in self.read_jsonl("tags.jsonl") if "chat_id" in r}

    def save_tags(self, tags: dict[str, dict[str, Any]]) -> None:
        self.write_jsonl("tags.jsonl", (tags[k] for k in sorted(tags)))

    def labels(self) -> dict[str, dict[str, Any]]:
        return {r["chat_id"]: r for r in self.read_jsonl("labels.jsonl") if "chat_id" in r}

    def save_labels(self, labels: dict[str, dict[str, Any]]) -> None:
        self.write_jsonl("labels.jsonl", (labels[k] for k in sorted(labels)))
