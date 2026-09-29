"""``install``: copy chosen generated skills into a skills directory, after confirmation."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .brief import skill_names
from .state import State

DEFAULT_TARGET = "~/.claude/skills"


class InstallError(RuntimeError):
    pass


@dataclass
class InstallPlan:
    target: Path
    copy: list[str] = field(default_factory=list)
    skip_existing: list[str] = field(default_factory=list)


def plan_install(state: State, names: Sequence[str], all_skills: bool, target: str, force: bool) -> InstallPlan:
    available = skill_names(state)
    if not available:
        raise InstallError("no generated skills; run `session-skills brief` first")
    if all_skills:
        chosen = available
    else:
        if not names:
            raise InstallError("name the skills to install, or pass --all. Available: " + ", ".join(available))
        unknown = [n for n in names if n not in available]
        if unknown:
            raise InstallError(f"unknown skill(s): {', '.join(unknown)}. Available: {', '.join(available)}")
        chosen = list(dict.fromkeys(names))
    plan = InstallPlan(Path(target).expanduser())
    for name in chosen:
        if (plan.target / name).exists() and not force:
            plan.skip_existing.append(name)
        else:
            plan.copy.append(name)
    return plan


def install(
    state: State,
    names: Sequence[str],
    *,
    all_skills: bool = False,
    target: str = DEFAULT_TARGET,
    force: bool = False,
    confirm: Callable[[InstallPlan], bool],
) -> InstallPlan:
    plan = plan_install(state, names, all_skills, target, force)
    if not plan.copy:
        return plan
    if not confirm(plan):
        raise InstallError("cancelled; nothing was copied")
    for name in plan.copy:
        src = state.root / "out" / "skills" / name
        dest = plan.target / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    return plan
