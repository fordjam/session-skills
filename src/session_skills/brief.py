"""``brief``: write the report, proposed skills, an AGENTS.md block and a CLAUDE.md snippet.

Everything goes to ``<state>/out``. Nothing is installed or edited outside the state directory;
``session-skills install`` copies chosen skills after confirmation.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .config import Config
from .questions import QUESTIONS, Question
from .redact import RedactOptions, short_quote
from .repeated import repeated as compute_repeated
from .state import State

BEGIN = "<!-- session-skills:begin -->"
END = "<!-- session-skills:end -->"
SELF_CHECK = "Before answering, ask: what would the user correct here? Fix that first."


@dataclass
class Pattern:
    question: Question
    chats: int
    pct: float
    last_seen: str
    threshold: float
    calibration: str
    quotes: list[str] = field(default_factory=list)


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _quotes(q: Question, chat_ids: list[str], chats: dict[str, dict[str, Any]], opts: RedactOptions, limit: int = 3) -> list[str]:
    """Short redacted human messages from flagged chats that contain one of the question's cues."""
    found: list[tuple[str, int, str]] = []
    for cid in chat_ids:
        chat = chats.get(cid, {})
        for message in chat.get("messages", []):
            text = str(message.get("text", ""))
            if message.get("role") != "user" or not text or len(text) > 400:
                continue
            lowered = text.lower()
            if any(cue in lowered for cue in q.cues):
                found.append((str(chat.get("date", "")), -len(text), text))
    found.sort(reverse=True)  # most recent first, then shortest
    quotes: list[str] = []
    for _, _, text in found:
        quote = short_quote(text, 120, opts)
        if quote and quote not in quotes:
            quotes.append(quote)
        if len(quotes) >= limit:
            break
    return quotes


def find_patterns(cfg: Config, state: State, today: date) -> tuple[list[Pattern], list[Pattern], list[str], int]:
    """Return (fresh patterns, stale patterns, excluded unreliable questions, tagged total)."""
    chats = state.chats()
    tags = {cid: t for cid, t in state.tags().items() if isinstance(t.get("scores"), dict)}
    calibration = state.read_json("calibration.json").get("questions", {})
    opts = cfg.redact_options()
    total = len(tags)
    cutoff = today - timedelta(days=cfg.stale_days)
    fresh: list[Pattern] = []
    stale: list[Pattern] = []
    excluded: list[str] = []
    for q in QUESTIONS:
        cal = calibration.get(q.name, {})
        status = str(cal.get("status", "uncalibrated"))
        if status == "unreliable":
            excluded.append(q.name)
            continue
        threshold = float(cal.get("threshold", cfg.default_threshold))
        flagged = [
            cid for cid, t in tags.items() if float(t["scores"].get(q.name, 0.0)) >= threshold
        ]
        if len(flagged) < cfg.min_chats:
            continue
        dates = sorted(str(chats.get(cid, {}).get("date") or tags[cid].get("date") or "") for cid in flagged)
        last_seen = dates[-1] if dates else ""
        flagged.sort(key=lambda cid: str(chats.get(cid, {}).get("date", "")), reverse=True)
        pattern = Pattern(
            q,
            len(flagged),
            round(100.0 * len(flagged) / max(total, 1), 1),
            last_seen,
            threshold,
            status,
            _quotes(q, flagged, chats, opts),
        )
        seen = _parse_date(last_seen)
        (stale if seen is None or seen < cutoff else fresh).append(pattern)
    fresh.sort(key=lambda p: (-p.chats, p.question.name))
    return fresh, stale, excluded, total


def render_skill(p: Pattern, total: int) -> str:
    q = p.question
    lines = [
        "---",
        f"name: {q.skill}",
        f"description: {json.dumps(q.description)}",
        "---",
        "",
        f"# {q.title}",
        "",
        "## When to use",
        "",
        q.description,
        "",
        "## What to do instead",
        "",
        *[f"- {step}" for step in q.instead],
        "",
        "## Why",
        "",
        f"{q.why} This came up in {p.chats} of {total} reviewed chats ({p.pct}%), most recently {p.last_seen}.",
        "",
        "## Examples",
        "",
    ]
    if p.quotes:
        lines += ["What the user said (redacted):", ""]
        lines += [f'- "{quote}"' for quote in p.quotes]
    else:
        lines.append("No short quote matched; see the report for counts.")
    lines.append("")
    return "\n".join(lines)


def render_rules(patterns: list[Pattern], today: date) -> list[str]:
    rules = [p for p in patterns if p.question.kind != "positive"]
    lines = [f"Learned from repeated corrections in past agent sessions (session-skills, {today.isoformat()}):", ""]
    lines += [f"- {p.question.rule}" for p in rules] or ["- (no repeated corrections found)"]
    lines += ["", SELF_CHECK]
    return lines


def render_report(
    cfg: Config,
    state: State,
    today: date,
    fresh: list[Pattern],
    stale: list[Pattern],
    excluded: list[str],
    total: int,
    repeated_report: dict[str, Any],
    skills: list[str],
) -> str:
    tracker = state.tracker()
    tags = state.tags()
    calibration = state.read_json("calibration.json")
    by_source = Counter(r.get("source", "") for r in tracker.values())
    failed = sum(1 for r in tracker.values() if r.get("status") == "failed")
    untagged = sum(1 for r in tracker.values() if r.get("status") == "new")
    out = [
        "# session-skills report",
        "",
        f"Generated {today.isoformat()}. All quotes are redacted. Nothing has been installed or changed.",
        "",
        f"- Chats found: {len(tracker)} ({', '.join(f'{k} {v}' for k, v in sorted(by_source.items())) or 'none'})",
        f"- Tagged: {total}; failed: {failed}; not yet tagged: {untagged}",
    ]
    if calibration:
        out.append(
            f"- Calibration: {calibration.get('labelled', 0)} chats re-labelled by `{calibration.get('labeller')}`; "
            f"questions excluded as unreliable: {', '.join(excluded) or 'none'}"
        )
    else:
        out.append("- Calibration: not run yet (`session-skills calibrate`); default threshold used for every question")
    out += ["", "## Repeated corrections", ""]
    corrections = [p for p in fresh if p.question.kind == "correction"]
    out += _pattern_table(corrections)
    out += ["", "## Workflow friction", ""]
    out += _pattern_table([p for p in fresh if p.question.kind == "workflow"])
    praise = [p for p in fresh if p.question.kind == "positive"]
    if praise:
        out += ["", "## What worked", ""]
        out += _pattern_table(praise)
    out += ["", "## Automation candidates (repeated asks)", ""]
    candidates = repeated_report.get("candidates", [])
    if candidates:
        out += [
            "| ask | example | sessions | times | first | last | idea |",
            "|---|---|---:|---:|---|---|---|",
        ]
        out += [
            f"| {c['ask']} | {c['example']} | {c['sessions']} | {c['occurrences']} | {c['first']} | {c['last']} | {c['suggestion']} |"
            for c in candidates[:25]
        ]
    else:
        out.append(f"No short ask recurred in {cfg.repeated_min_sessions} or more sessions.")
    out += ["", "## Tasks and outcomes", ""]
    out += _task_table(tags)
    if calibration.get("questions"):
        out += ["", "## Calibration", "", "| question | agreement | kappa | threshold | status |", "|---|---:|---:|---:|---|"]
        for name, q in calibration["questions"].items():
            kappa = "" if q.get("kappa") is None else q["kappa"]
            out.append(f"| {name} | {q.get('agreement')} | {kappa} | {q.get('threshold')} | {q.get('status')} |")
    if stale:
        out += ["", f"Dropped as stale (not seen in the last {cfg.stale_days} days): " + ", ".join(p.question.name for p in stale)]
    out += ["", "## Proposed files", ""]
    if skills:
        out += [f"- `out/skills/{name}/SKILL.md`" for name in skills]
    out += [
        "- `out/AGENTS.md` (block for OpenCode / Codex `AGENTS.md`)",
        "- `out/CLAUDE.md.snippet` (lines for `CLAUDE.md`)",
        "",
        "Review them, then `session-skills install` to copy chosen skills (asks first).",
        "",
        SELF_CHECK,
        "",
    ]
    return "\n".join(out)


def _pattern_table(patterns: list[Pattern]) -> list[str]:
    if not patterns:
        return ["None above the minimum count."]
    rows = ["| pattern | chats | % | last seen | quote |", "|---|---:|---:|---|---|"]
    for p in patterns:
        quote = f'"{p.quotes[0]}"' if p.quotes else "—"
        rows.append(f"| {p.question.name} | {p.chats} | {p.pct}% | {p.last_seen} | {quote} |")
    return rows


def _task_table(tags: dict[str, dict[str, Any]]) -> list[str]:
    if not tags:
        return ["No tagged chats."]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in tags.values():
        groups[str(t.get("task", "other"))].append(t)
    rows = ["| task | chats | resolved | mean frustration |", "|---|---:|---:|---:|"]
    for task, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        resolved = sum(1 for t in items if t.get("outcome") == "resolved")
        levels = [int(t["frustration"]) for t in items if t.get("frustration")]
        mean = f"{sum(levels) / len(levels):.1f}" if levels else "—"
        rows.append(f"| {task} | {len(items)} | {resolved} | {mean} |")
    return rows


def brief(cfg: Config, state: State, today: date | None = None) -> dict[str, Any]:
    today = today or date.today()
    fresh, stale, excluded, total = find_patterns(cfg, state, today)
    repeated_report = compute_repeated(cfg, state)  # cheap and deterministic: always fresh
    out = state.out
    skills_dir = out / "skills"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)
    written: list[str] = []
    for p in fresh:
        if p.question.kind == "positive":
            continue
        state.write_text(skills_dir / p.question.skill / "SKILL.md", render_skill(p, total))
        written.append(p.question.skill)
    rules = render_rules(fresh, today)
    state.write_text(out / "AGENTS.md", "\n".join([BEGIN, "## Working with this user", "", *rules, END, ""]))
    state.write_text(out / "CLAUDE.md.snippet", "\n".join(["## Working with me", "", *rules, ""]))
    report = render_report(cfg, state, today, fresh, stale, excluded, total, repeated_report, written)
    report_path = state.write_text(out / "report.md", report)
    return {
        "report": str(report_path),
        "patterns": [p.question.name for p in fresh],
        "stale": [p.question.name for p in stale],
        "excluded": excluded,
        "skills": written,
        "automation_candidates": len(repeated_report.get("candidates", [])),
    }


def skill_names(state: State) -> list[str]:
    root: Path = state.root / "out" / "skills"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "SKILL.md").is_file())

