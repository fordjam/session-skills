"""``install`` / ``rollback``: put chosen generated files where an agent will read them.

Destinations by agent (``--agent``):

  claude    skills -> $CLAUDE_CONFIG_DIR/skills (default ~/.claude/skills)
            rules  -> $CLAUDE_CONFIG_DIR/CLAUDE.md
  opencode  skills -> $XDG_CONFIG_HOME/opencode/skills (default ~/.config/opencode/skills)
            rules  -> $XDG_CONFIG_HOME/opencode/AGENTS.md
  codex     skills -> ~/.agents/skills
            rules  -> $CODEX_HOME/AGENTS.md (default ~/.codex/AGENTS.md)

Rules are written only with ``--rules``, as a block between ``<!-- session-skills:begin -->`` and
``<!-- session-skills:end -->``; everything else in the file is preserved and a later install
replaces only that block. Every install shows a unified diff of each file first, asks, and saves
the previous contents under ``<state>/backups/<id>`` so ``session-skills rollback`` can undo it.
"""

from __future__ import annotations

import contextlib
import difflib
import json
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .brief import BEGIN, END, skill_names
from .state import State

AGENTS = ("claude", "opencode", "codex")
_SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class InstallError(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentTarget:
    agent: str
    skills_dir: Path
    rules_file: Path


def agent_target(agent: str, skills_dir: str | None = None) -> AgentTarget:
    home = Path.home()
    if agent == "claude":
        root = Path(os.environ.get("CLAUDE_CONFIG_DIR") or home / ".claude").expanduser()
        target = AgentTarget(agent, root / "skills", root / "CLAUDE.md")
    elif agent == "opencode":
        root = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config").expanduser() / "opencode"
        target = AgentTarget(agent, root / "skills", root / "AGENTS.md")
    elif agent == "codex":
        root = Path(os.environ.get("CODEX_HOME") or home / ".codex").expanduser()
        target = AgentTarget(agent, home / ".agents" / "skills", root / "AGENTS.md")
    else:
        raise InstallError(f"unknown agent {agent!r}; choose from {', '.join(AGENTS)}")
    if skills_dir:
        target = AgentTarget(agent, Path(skills_dir).expanduser(), target.rules_file)
    return target


@dataclass
class Change:
    path: Path
    old: str | None  # None: the file does not exist yet
    new: str
    label: str

    def diff(self) -> str:
        old_lines = (self.old or "").splitlines(keepends=True)
        new_lines = self.new.splitlines(keepends=True)
        before = str(self.path) if self.old is not None else "/dev/null"
        return "".join(difflib.unified_diff(old_lines, new_lines, before, str(self.path)))


@dataclass
class InstallPlan:
    target: AgentTarget
    changes: list[Change] = field(default_factory=list)
    copy: list[str] = field(default_factory=list)  # skill names that will be written
    skip_existing: list[str] = field(default_factory=list)  # differ, need --force
    unchanged: list[str] = field(default_factory=list)
    rules: bool = False


def merge_block(existing: str | None, block: str) -> str:
    """Put ``block`` (which starts with BEGIN and ends with END) into ``existing``, replacing a
    previous block and leaving everything else untouched."""
    text = existing or ""
    start, end = text.find(BEGIN), text.find(END)
    if start != -1 and end > start:
        return text[:start] + block + text[end + len(END) :]
    if not text:
        return block + "\n"
    return text.rstrip("\n") + "\n\n" + block + "\n"


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def rules_block(state: State, agent: str) -> str:
    out = state.root / "out"
    if agent == "claude":
        body = _read(out / "CLAUDE.md.snippet")
        if body is None:
            raise InstallError("no generated rules; run `session-skills brief` first")
        return f"{BEGIN}\n{body.strip()}\n{END}"
    block = _read(out / "AGENTS.md")
    if block is None:
        raise InstallError("no generated rules; run `session-skills brief` first")
    return block.strip()


def plan_install(
    state: State,
    names: Sequence[str],
    all_skills: bool,
    *,
    agent: str = "claude",
    target: str | None = None,
    force: bool = False,
    rules: bool = False,
) -> InstallPlan:
    where = agent_target(agent, target)
    available = skill_names(state)
    if all_skills:
        chosen = available
    elif names:
        unknown = [n for n in names if n not in available]
        if unknown:
            raise InstallError(f"unknown skill(s): {', '.join(unknown)}. Available: {', '.join(available) or 'none'}")
        chosen = list(dict.fromkeys(names))
    elif rules:
        chosen = []
    else:
        if not available:
            raise InstallError("no generated skills; run `session-skills brief` first")
        listed = ", ".join(available)
        raise InstallError(f"name the skills to install, pass --all, or pass --rules. Available: {listed}")
    if all_skills and not available and not rules:
        raise InstallError("no generated skills; run `session-skills brief` first")
    plan = InstallPlan(where, rules=rules)
    for name in chosen:
        if not _SKILL_NAME.match(name):
            raise InstallError(f"refusing unexpected skill name {name!r}")
        new = (state.root / "out" / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
        dest = where.skills_dir / name / "SKILL.md"
        old = _read(dest)
        if old == new:
            plan.unchanged.append(name)
        elif (old is not None or (where.skills_dir / name).exists()) and not force:
            plan.skip_existing.append(name)
        else:
            plan.copy.append(name)
            plan.changes.append(Change(dest, old, new, f"skill {name}"))
    if rules:
        old = _read(where.rules_file)
        new = merge_block(old, rules_block(state, agent))
        if new != old:
            plan.changes.append(Change(where.rules_file, old, new, f"rules block in {where.rules_file.name}"))
    return plan


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        if path.exists():
            os.chmod(tmp, path.stat().st_mode & 0o777)
        else:
            os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _backup(state: State, changes: list[Change]) -> str:
    backup_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")  # noqa: DTZ005 - local, only an id
    root = state.root / "backups" / backup_id
    entries = []
    for index, change in enumerate(changes):
        entry: dict[str, object] = {"path": str(change.path), "existed": change.old is not None}
        if change.old is not None:
            state.write_text(root / f"{index}.orig", change.old)
            entry["backup"] = f"{index}.orig"
        created_dirs = []
        parent = change.path.parent
        while not parent.exists():
            created_dirs.append(str(parent))
            parent = parent.parent
        entry["created_dirs"] = created_dirs
        entries.append(entry)
    state.write_text(root / "manifest.json", json.dumps({"id": backup_id, "files": entries}, indent=2))
    return backup_id


def install(
    state: State,
    names: Sequence[str],
    *,
    all_skills: bool = False,
    agent: str = "claude",
    target: str | None = None,
    force: bool = False,
    rules: bool = False,
    confirm: Callable[[InstallPlan], bool],
) -> tuple[InstallPlan, str]:
    """Show, confirm, back up, write. Returns (plan, backup id or "")."""
    state.lock()
    plan = plan_install(state, names, all_skills, agent=agent, target=target, force=force, rules=rules)
    if not plan.changes:
        return plan, ""
    if not confirm(plan):
        raise InstallError("cancelled; nothing was changed")
    backup_id = _backup(state, plan.changes)
    try:
        for change in plan.changes:
            _atomic_write(change.path, change.new)
    except BaseException:
        rollback(state, backup_id)
        raise
    return plan, backup_id


def backups(state: State) -> list[str]:
    root = state.root / "backups"
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir() if (p / "manifest.json").is_file() and not p.name.endswith(".rolled-back")
    )


def rollback(state: State, backup_id: str | None = None) -> list[str]:
    """Undo one install (the latest by default): restore changed files, remove created ones.
    Returns the paths touched."""
    state.lock()
    available = backups(state)
    if not available:
        raise InstallError("nothing to roll back")
    chosen = backup_id or available[-1]
    if chosen not in available:
        raise InstallError(f"unknown backup {chosen!r}; available: {', '.join(available)}")
    root = state.root / "backups" / chosen
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    touched = []
    for entry in reversed(manifest["files"]):
        path = Path(entry["path"])
        if entry["existed"]:
            _atomic_write(path, (root / entry["backup"]).read_text(encoding="utf-8"))
        else:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            for directory in entry.get("created_dirs", []):
                with contextlib.suppress(OSError):
                    Path(directory).rmdir()  # only if empty
        touched.append(str(path))
    root.rename(root.with_name(chosen + ".rolled-back"))
    return touched
