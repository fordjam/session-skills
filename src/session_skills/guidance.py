"""Shape check for files an agent will load as instructions.

Installed files (``SKILL.md``, the ``AGENTS.md`` block, the ``CLAUDE.md`` snippet) are built only
from this package's fixed guidance text plus counts and dates. They never contain user text:
quotes live in ``out/report.md`` and ``out/evidence/``, which are never installed.

As a second, independent guard, every line of an installed file must fit an allow-list shape:
guidance about response style and working process. A line fails if it contains a URL, a path, a
file name, a command-line flag, shell syntax, a credential word or a network/transfer verb.
``brief`` refuses to write, and ``install`` refuses to copy, a file that fails.
"""

from __future__ import annotations

import re

_EXEMPT = {"<!-- session-skills:begin -->", "<!-- session-skills:end -->", "---"}

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("url", r"\b[a-z][a-z0-9+.-]*://|\bwww\.|\b[a-z0-9-]+\.(?:com|net|org|io|dev|test|example)\b"),
        ("path", r"\S/\S|~|\\|(?<![\w])\.{1,2}/"),
        ("file name", r"\b[a-z_][\w-]*\.[a-z][a-z0-9]{1,4}\b"),
        ("command flag", r"(?<!\S)--?[a-z]"),
        ("shell syntax", r"[`$|<>{}]|&&"),
        ("credential", r"\b(password|passwd|passphrase|secret|token|api[ _-]?keys?|private[ _-]?keys?|key ?files?|"
         r"id_rsa|id_ed25519|ssh|credentials?|keychain|cookies?|env(?:ironment)? var\w*)\b"),
        ("network verb", r"\b(upload\w*|download\w*|exfiltrat\w*|transmit\w*|curl|wget|scp|rsync|ftp|https?|"
         r"webhook\w*|e-?mail\w*|server\w*|endpoints?|beacon\w*)\b|\bsend\b[^.]{0,40}\bto\b|\bpost\w*\b[^.]{0,20}\bto\b"),
        ("command", r"\b(sudo|rm|chmod|chown|bash|zsh|powershell|eval|exec|base64|nc|netcat|python|node)\b"),
        ("data disclosure", r"\b(contents? of|include|paste|attach|reveal|print|dump|share|copy|read out)\b[^.]{0,40}"
         r"\b(files?|config\w*|keys?|logs?|history|environment|director(?:y|ies)|folders?|home)\b"),
        ("role or override", r"\b(system|assistant|developer)\s*:|\byou are now\b|\bdeveloper mode\b|"
         r"\b(ignore|disregard|forget|override)\b[^.]{0,40}\b(instructions?|previous|above|rules?)\b"),
    )
)


def problems(text: str) -> list[str]:
    """Return one message per offending line; empty when the text fits the allowed shape."""
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped in _EXEMPT:
            continue
        if stripped.startswith(("name:", "description:")) and number <= 4:
            stripped = stripped.split(":", 1)[1]
        for name, pattern in _RULES:
            if pattern.search(stripped):
                found.append(f"line {number}: {name}: {stripped[:80]}")
                break
    return found


class UnsafeGuidance(ValueError):
    pass


def check(text: str, where: str) -> str:
    """Return ``text`` unchanged, or raise when any line fails the shape check."""
    issues = problems(text)
    if issues:
        raise UnsafeGuidance(f"refusing to write {where}: " + "; ".join(issues))
    return text
