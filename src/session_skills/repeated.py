"""``repeated``: deterministic clustering of short human messages across sessions (no model).

Short asks that recur across many sessions ("run the tests", "status?", "commit and push") are
automation candidates: a hook, a slash command, a script or a permission rule would remove them.

Clustering is negation-aware: "run the tests" and "never run the tests" share every content word
but mean opposite things, so a message's polarity is part of its cluster key. Text is redacted
again with the current policy before anything is counted or shown.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .collect import refresh_redaction
from .config import Config
from .redact import redact, short_quote
from .state import State

STOPWORDS = frozenset(
    """a an and are as at be but by can could do does for from hey hi i i'm im in is it it's its
    just let me my of on or our please pls plz so that the then this to u us we would you your
    yours kindly now also really maybe quick quickly real""".split()  # noqa: SIM905
)
NEGATORS = frozenset(
    """not no never don't dont doesn't doesnt didn't didnt won't wont can't cant cannot shouldn't
    shouldnt stop without avoid quit nope""".split()  # noqa: SIM905
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

_WORDS = str.split
SUGGESTIONS: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset(_WORDS("status progress running done left update where eta stuck idle")),
     "status digest: a hook or scheduled summary"),
    (frozenset(_WORDS("commit push merge pr rebase branch land tag")), "slash command or script for the git flow"),
    (frozenset(_WORDS("test tests lint build typecheck ci format")),
     "hook that runs the check automatically after edits"),
    (frozenset(_WORDS("yes ok okay continue proceed go approve approved sure yep")),
     "check what was being approved each time; if it is one routine action, a narrowly scoped permission rule"),
    (frozenset(_WORDS("deploy release publish ship")), "release script with one confirmation step"),
)  # fmt: skip
DEFAULT_SUGGESTION = "custom slash command or skill"
NEGATIVE_SUGGESTION = "standing instruction (AGENTS.md / CLAUDE.md) or a hook that blocks it"


def normalise(text: str) -> tuple[str, ...]:
    value = _PLACEHOLDER.sub(" secret ", text)
    value = _URL.sub(" url ", value)
    value = _CODE.sub(" code ", value)
    value = _PATH.sub(" path ", value)
    value = value.lower()
    value = _ID.sub(" id ", value)
    value = _NUM.sub(" n ", value)
    value = re.sub(r"[^a-z' ]+", " ", value)
    tokens = [t.strip("'") if t.strip("'") not in NEGATORS else t for t in value.split()]
    return tuple(t for t in tokens if t and t not in STOPWORDS)


def polarity(tokens: tuple[str, ...]) -> str:
    """"neg" when the ask forbids something ("don't run tests"), else "pos"."""
    return "neg" if any(t in NEGATORS or t.strip("'") in NEGATORS for t in tokens) else "pos"


def content(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(t for t in tokens if t not in NEGATORS and t.strip("'") not in NEGATORS)


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
    polarity: str = "pos"
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
        words = content(tokens)
        if not words:
            continue
        sign = polarity(tokens)
        key = (sign, *sorted(set(words)))
        target = exact.get(key)
        if target is None:
            tokenset = frozenset(words)
            candidates = {id(c): c for t in tokenset for c in by_token.get(f"{sign}:{t}", [])}
            scored = sorted(
                ((jaccard(tokenset, c.seed), -len(c.chats), c.order, c) for c in candidates.values()),
                key=lambda s: (-s[0], s[1], s[2]),
            )
            if scored and scored[0][0] >= SIMILARITY:
                target = scored[0][3]
            else:
                target = Cluster(tokenset, len(clusters), sign)
                clusters.append(target)
                for t in tokenset:
                    by_token.setdefault(f"{sign}:{t}", []).append(target)
            exact[key] = target
        target.forms[" ".join(tokens)] += 1
        target.chats.add(cid)
        if date:
            target.dates.append(date)
        target.examples.append(text)
        target.occurrences += 1
    return clusters


def repeated(cfg: Config, state: State) -> dict[str, Any]:
    refresh_redaction(cfg, state)
    opts = cfg.redact_options()
    items = [
        # the message's own date: an ask made today in an old session counts as today
        (
            cid,
            str(m.get("date") or chat.get("last_date") or chat.get("date") or ""),
            redact(str(m.get("text", "")), opts),
        )
        for cid, chat in state.chats().items()
        for m in chat.get("messages", [])
        if m.get("role") == "user" and 0 < len(str(m.get("text", ""))) <= cfg.repeated_max_chars
    ]
    rows: list[dict[str, Any]] = []
    for cluster in cluster_messages(items):
        if len(cluster.chats) < cfg.repeated_min_sessions:
            continue
        label = min(cluster.forms.items(), key=lambda kv: (-kv[1], kv[0]))[0]
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
                "polarity": cluster.polarity,
                "suggestion": NEGATIVE_SUGGESTION if cluster.polarity == "neg" else suggestion(cluster.seed),
            }
        )
    rows.sort(key=lambda r: (-r["sessions"], -r["occurrences"], r["ask"]))
    report = {"messages_considered": len(items), "candidates": rows}
    state.write_json("repeated.json", report)
    return report
