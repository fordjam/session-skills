"""``tag``: one cheap-model request per chat (per 90k-char part), JSON answers, resumable."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .backends import Backend, BackendError
from .collect import render, split_parts
from .config import Config
from .questions import OUTCOMES, QUESTIONS, QUESTION_NAMES, TASKS
from .sources.base import TOOL_MARKER
from .state import State


class AnswerError(ValueError):
    pass


def build_prompt(part: str, index: int = 1, total: int = 1) -> str:
    questions = "\n".join(f"- {q.name}: {q.ask}" for q in QUESTIONS)
    return (
        f"{TOOL_MARKER}\n"
        "You are labelling ONE conversation between a user and an AI coding assistant.\n"
        "Reply with exactly one JSON object and nothing else (no prose, no markdown).\n"
        "Keys:\n"
        "- one key per question below: a probability from 0 to 1 that it happened here\n"
        f"- task: exactly one of {list(TASKS)}\n"
        f"- outcome: exactly one of {list(OUTCOMES)}\n"
        "- frustration: integer 1-5, the user's peak frustration (1 calm, 5 very frustrated)\n"
        "Questions:\n"
        f"{questions}\n"
        "Text is redacted: placeholders such as [REDACTED:key] or [PASTED BLOCK: 40 lines] are not errors.\n"
        "Do not use tools. Judge only from the text.\n"
        f"This is part {index} of {total} of the conversation.\n"
        "CONVERSATION:\n"
        f"{part}\n"
    )


def _json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    found = []
    cleaned = re.sub(r"```(?:json)?", "", text)
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except ValueError:
            continue
        if isinstance(value, dict):
            found.append(value)
    return found


def _probability(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip().rstrip("%")) / (100 if value.strip().endswith("%") else 1)
        except ValueError:
            return None
    else:
        return None
    return number if 0.0 <= number <= 1.0 else None


def parse_answer(text: str) -> dict[str, Any]:
    """Validate a model reply. Every question must have a probability; otherwise it is an error
    (and the chat stays blank rather than being recorded as 0)."""
    candidates = [o for o in _json_objects(text) if sum(1 for n in QUESTION_NAMES if n in o) >= len(QUESTION_NAMES) // 2]
    if not candidates:
        raise AnswerError("no JSON object with question keys")
    obj = candidates[0]
    scores: dict[str, float] = {}
    for name in QUESTION_NAMES:
        value = _probability(obj.get(name))
        if value is None:
            raise AnswerError(f"missing or invalid score for {name}")
        scores[name] = value
    task = str(obj.get("task", "")).strip().lower()
    outcome = str(obj.get("outcome", "")).strip().lower()
    frustration: int | None
    try:
        frustration = min(5, max(1, int(round(float(obj.get("frustration"))))))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        frustration = None
    return {
        "scores": scores,
        "task": task if task in TASKS else "other",
        "outcome": outcome if outcome in OUTCOMES else "unclear",
        "frustration": frustration,
    }


def merge(answers: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-part answers: max per question, first part's task, last part's outcome,
    peak frustration."""
    scores = {n: max(a["scores"][n] for a in answers) for n in QUESTION_NAMES}
    frustrations = [a["frustration"] for a in answers if a.get("frustration")]
    return {
        "scores": scores,
        "task": answers[0]["task"],
        "outcome": answers[-1]["outcome"],
        "frustration": max(frustrations) if frustrations else None,
    }


def ask(backend: Backend, prompt: str, retries: int) -> dict[str, Any]:
    last: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            return parse_answer(backend.complete(prompt))
        except (BackendError, AnswerError, OSError) as exc:
            last = exc
    raise AnswerError(str(last) if last else "no attempts")


def label_chat(chat: dict[str, Any], backend: Backend, cfg: Config) -> tuple[dict[str, Any], int]:
    """Return (merged answer, requests used) for one chat."""
    parts = split_parts(render(chat.get("messages", [])), cfg.max_chars) or [""]
    answers = [ask(backend, build_prompt(p, i, len(parts)), cfg.retries) for i, p in enumerate(parts, 1)]
    return merge(answers), len(parts)


@dataclass
class TagSummary:
    tagged: int = 0
    failed: int = 0
    requests: int = 0
    remaining: int = 0
    dry_run: bool = False
    planned_chats: int = 0
    planned_chars: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "tagged": self.tagged,
            "failed": self.failed,
            "requests": self.requests,
            "remaining": self.remaining,
        }
        if self.dry_run:
            data.update(
                dry_run=True,
                planned_chats=self.planned_chats,
                planned_requests=self.requests,
                estimated_input_tokens=self.planned_chars // 4 + self.requests * 600,
            )
        if self.errors:
            data["errors"] = self.errors
        return data


def pending(tracker: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    """Untagged chats: new ones first, then earlier failures; newest first within each group."""

    def group(status: str) -> list[dict[str, str]]:
        rows = [r for r in tracker.values() if r.get("status") == status]
        return sorted(rows, key=lambda r: (r.get("date") or "", r["chat_id"]), reverse=True)

    return group("new") + group("failed")


def tag(
    cfg: Config,
    state: State,
    backend: Backend,
    *,
    dry_run: bool = False,
    progress: Callable[[str], None] | None = None,
) -> TagSummary:
    tracker = state.tracker()
    chats = state.chats()
    tags = state.tags()
    summary = TagSummary(dry_run=dry_run)
    queue = pending(tracker)
    budget = cfg.max_requests
    for row in queue:
        need = max(1, int(row.get("parts") or 1))
        if need > budget and summary.requests > 0:
            break  # leave it for the next run
        chat = chats.get(row["chat_id"])
        if dry_run:
            summary.planned_chats += 1
            summary.planned_chars += int(row.get("chars") or 0)
            summary.requests += need
            budget -= need
            continue
        if chat is None:
            row.update(status="failed", flagged="", note="chat missing from chats.jsonl; rerun build-list")
            summary.failed += 1
            continue
        try:
            answer, used = label_chat(chat, backend, cfg)
        except AnswerError as exc:
            used = need
            row.update(status="failed", flagged="", note=f"failed after {cfg.retries} tries")
            key = str(exc).split(":")[0][:60]
            summary.errors[key] = summary.errors.get(key, 0) + 1
            tags.pop(row["chat_id"], None)
            summary.failed += 1
        else:
            flagged = any(v >= cfg.default_threshold for v in answer["scores"].values())
            row.update(status="tagged", flagged="1" if flagged else "0", note=f"task={answer['task']}")
            tags[row["chat_id"]] = {
                "chat_id": row["chat_id"],
                "date": chat.get("date", ""),
                "backend": backend.spec,
                "tagged_at": datetime.now(timezone.utc).date().isoformat(),
                **answer,
            }
            summary.tagged += 1
        summary.requests += used
        budget -= used
        state.save_tracker(tracker)
        state.save_tags(tags)
        if progress:
            progress(f"{summary.tagged + summary.failed}/{len(queue)} chats, {summary.requests} requests")
    summary.remaining = len(queue) - summary.planned_chats if dry_run else len(pending(tracker))
    return summary
