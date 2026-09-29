"""Make user quotes safe to embed in files an agent will later read as instructions.

Generated ``SKILL.md`` files are loaded by coding agents. A quote copied verbatim from a
transcript could smuggle in instructions ("ignore previous instructions and run ..."), a
heading that looks like a new section, a code block or a link. Every quote that reaches a
generated file goes through :func:`sanitize_quote`:

* redacted again with the current policy;
* flattened to one line of plain text: no headings, list or quote markers, code fences,
  backticks, links, images, HTML or table pipes;
* dropped entirely when it looks like an injected instruction (ignore/disregard previous
  instructions, run/execute this, piped shell, role markers, ...);
* capped in length and stripped of double quotes so it cannot close the quote it sits in.

Callers then wrap it in an explicit "user quote, not an instruction" label.
"""

from __future__ import annotations

import re

from .redact import DEFAULT_OPTIONS, RedactOptions, redact

MAX_QUOTE = 120

_INJECTION = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(ignore|disregard|forget|override|bypass|overrule)\b.{0,60}?\b(instructions?|rules?|prompts?|guidelines|"
        r"previous|above|earlier|system|everything|all)\b",
        r"\b(run|execute|exec|eval|invoke)\b\s*(this|these|that|the following|following|:|`|\$|the command|"
        r"this command|command|script|code)",
        r"\b(curl|wget|iwr|invoke-webrequest)\b.{0,80}?(https?://|\|)",
        r"\|\s*(sh|bash|zsh|python\d?|node|perl|ruby)\b",
        r"\bsudo\s|\brm\s+-[a-z]*r[a-z]*f|\bchmod\s+[0-7+]|\bbase64\s+(-d|--decode)",
        r"\b(you are now|new instructions|system prompt|developer mode|jailbreak|from now on you)\b",
        r"(^|\s)(system|assistant|developer|user)\s*:",
        r"<\|[a-z_]*\|>|\[/?(inst|sys)\]",
        r"\b(always|never)\b.{0,40}?\b(push|deploy|delete|approve|commit|merge|send|upload)\b"
        r".{0,40}?\b(without|no matter)\b",
    )
)
_FENCE = re.compile(r"```|~~~")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_REF_LINK = re.compile(r"\[([^\]]*)\]\[[^\]]*\]")
_URL = re.compile(r"\b(?:https?|ftp|file|data|javascript):\S+", re.IGNORECASE)
_HTML_COMMENT = re.compile(r"<!--.*?(-->|$)", re.DOTALL)
_HTML = re.compile(r"</?[A-Za-z][^>]*>")
_LINE_MARKERS = re.compile(r"(?m)^\s*(#{1,6}\s*|>+\s*|[-*+]\s+|\d+[.)]\s+|={3,}|-{3,})")
_PLACEHOLDER = re.compile(r"(\[REDACTED:[a-z-]+\]|\[PASTED BLOCK(?:: \d+ lines)?\])")
_UNSAFE_CHARS = re.compile(r"[`#<>|*\[\]{}\"\\]")


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION)


def sanitize_quote(text: str, opts: RedactOptions = DEFAULT_OPTIONS, limit: int = MAX_QUOTE) -> str | None:
    """Return a safe one-line plain-text quote, or ``None`` when the quote must not be used."""
    value = redact(text, opts)
    if looks_like_injection(value):
        return None
    value = _HTML_COMMENT.sub(" ", value)
    value = _FENCE.sub(" ", value)
    value = _IMAGE.sub(" ", value)
    value = _LINK.sub(r"\1", value)
    value = _REF_LINK.sub(r"\1", value)
    value = _URL.sub("(link)", value)
    value = _HTML.sub(" ", value)
    value = _LINE_MARKERS.sub(" ", value)
    # keep redaction placeholders verbatim; strip markup characters everywhere else
    pieces = _PLACEHOLDER.split(value)
    value = "".join(piece if _PLACEHOLDER.fullmatch(piece) else _UNSAFE_CHARS.sub(" ", piece) for piece in pieces)
    value = " ".join(value.split())
    if not value or looks_like_injection(value):
        return None
    if len(value) > limit:
        value = value[: limit - 1].rstrip() + "…"
    return value
