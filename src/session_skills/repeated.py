"""``repeated``: deterministic clustering of short human messages across sessions (no model).

Short asks that recur across many sessions ("run the tests", "status?", "commit and push") are
automation candidates: a hook, a slash command, a script or a permission rule would remove them.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .redact import short_quote
from .state import State

STOPWORDS = frozenset(
    """a an and are as at be but by can could do does for from hey hi i i'm im in is it it's its
    just let me my of on or our please pls plz so that the then this to u us we would you your
    yours kindly now also really maybe quick quickly real""".split()
)
SIMILARITY = 0.5

_PLACEHOLDER = re.compile(r"\[(?:REDACTED:[a-z-]+|PASTED BLOCK[^\]]*)\]")
_URL = re.compile(r"https?://\S+")
_PATH = re.compile(
    r"(?<!\S)[\w.~@-]*(?:/[\w.@-]+)+/?|\b[\w-]+\.(?:py|ts|js|tsx|md|json|toml|yml|yaml|go|rs|sh|txt)\b"
)
_CODE = re.compile(r"`[^`]*`")
_ID = re.compile(r"\b[0-9a-f]{7,}\b")
_NUM = re.compile(r"\b\d+(?:\.\d+)*\b")

SUGGESTIONS: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset("status progress running done left update where eta stuck idle".split()), "status digest: a hook or scheduled summary"),
    (frozenset("commit push merge pr rebase branch land tag".split()), "slash command or script for the git flow"),
    (frozenset("test tests lint build typecheck ci format".split()), "hook that runs the check automatically after edits"),
    (frozenset("yes ok okay continue proceed go approve approved sure yep".split()), "permission allowlist or auto-approve rule"),
    (frozenset("deploy release publish ship".split()), "release script with one confirmation step"),
)
DEFAULT_SUGGESTION = "custom slash command or skill"


def normalise(text: str) -> tuple[str, ...]:
    value = _PLACEHOLDER.sub(" secret ", text)
    value = _URL.sub(" url ", value)
    value = _CODE.sub(" code ", value)
    value = _PATH.sub(" path ", value)
    value = value.lower()
    value = _ID.sub(" id ", value)
    value = _NUM.sub(" n ", value)
    value = re.sub(r"[^a-z' ]+", " ", value)
    tokens = [t.strip("'") for t in value.split()]
    return tuple(t for t in tokens if t and t not in STOPWORDS)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def suggestion(tokens: frozenset[str]) -> str:
    best, best_hits = DEFAULT_SUGGESTION, 0
    for words, text in SUGGESTIONS:
        hits = len(tokens & words)
        if hits > best_hits:
            best, best_hits = text, hits
    return best


@dataclass
class Cluster:
    seed: frozenset[str]
    order: int = 0
    forms: Counter[str] = field(default_factory=Counter)
    chats: set[str] = field(default_factory=set)
    dates: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    occurrences: int = 0


def cluster_messages(items: list[tuple[str, str, str]]) -> list[Cluster]:
    """``items`` are (chat_id, date, redacted text). Greedy clustering over a fixed sort order,
    so the result is deterministic."""
    clusters: list[Cluster] = []
    exact: dict[tuple[str, ...], Cluster] = {}
    by_token: dict[str, list[Cluster]] = {}
    for cid, date, text in sorted(items, key=lambda t: (t[1], t[0], t[2])):
        tokens = normalise(text)
        if not tokens:
            continue
        key = tuple(sorted(set(tokens)))
        target = exact.get(key)
        if target is None:
            tokenset = frozenset(key)
            candidates = {id(c): c for t in tokenset for c in by_token.get(t, [])}
            scored = sorted(
                ((jaccard(tokenset, c.seed), -len(c.chats), c.order, c) for c in candidates.values()),
                key=lambda s: (-s[0], s[1], s[2]),
            )
            if scored and scored[0][0] >= SIMILARITY:
                target = scored[0][3]
            else:
                target = Cluster(tokenset, len(clusters))
                clusters.append(target)
                for t in tokenset:
                    by_token.setdefault(t, []).append(target)
            exact[key] = target
        target.forms[" ".join(tokens)] += 1
        target.chats.add(cid)
        if date:
            target.dates.append(date)
        target.examples.append(text)
        target.occurrences += 1
    return clusters


def repeated(cfg: Config, state: State) -> dict[str, Any]:
    opts = cfg.redact_options()
    items = [
        (cid, str(chat.get("date") or m.get("date") or ""), str(m.get("text", "")))
        for cid, chat in state.chats().items()
        for m in chat.get("messages", [])
        if m.get("role") == "user" and 0 < len(str(m.get("text", ""))) <= cfg.repeated_max_chars
    ]
    rows: list[dict[str, Any]] = []
    for cluster in cluster_messages(items):
        if len(cluster.chats) < cfg.repeated_min_sessions:
            continue
        label = sorted(cluster.forms.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        dates = sorted(cluster.dates)
        example = min(cluster.examples, key=lambda e: (len(e), e))
        rows.append(
            {
                "ask": label,
                "example": short_quote(example, 80, opts),
                "sessions": len(cluster.chats),
                "occurrences": cluster.occurrences,
                "first": dates[0] if dates else "",
                "last": dates[-1] if dates else "",
                "suggestion": suggestion(cluster.seed),
            }
        )
    rows.sort(key=lambda r: (-r["sessions"], -r["occurrences"], r["ask"]))
    report = {"messages_considered": len(items), "candidates": rows}
    state.write_json("repeated.json", report)
    return report
