"""Deterministic redaction applied to every piece of text before it is stored or sent to a model.

The redactor is intentionally conservative: it prefers removing too much over leaking. It is
pure (no I/O, no randomness) and idempotent: ``redact(redact(x)) == redact(x)``.

What it removes, in order:

1. PEM private-key blocks.
2. Pasted blocks: ``<pasted_content>`` tags, fenced code blocks longer than ``max_block_lines``,
   and unfenced runs of ``max_block_lines`` or more consecutive non-empty lines.
3. ``.env`` contents: fenced env/dotenv blocks, ``NAME=value`` lines whose name looks secret, and
   every ``NAME=value`` line in a message that mentions ``.env``.
4. Credentials in URLs (``scheme://user:pass@``) and secret-looking query parameters.
5. API keys and tokens: well-known prefixes, JWTs, ``Bearer``/``Basic`` values, and generic
   high-entropy strings (32+ chars mixing upper, lower and digits, or 32+ hex digits).
6. Email addresses.
7. Home-directory user names (``/Users/<name>``, ``/home/<name>``, ``C:\\Users\\<name>`` -> ``~``).
8. Money amounts (``$1,234.50`` -> ``$N``; also GBP and EUR symbols).
9. Runs of 12-19 digits, optionally separated by spaces or dashes (cards, accounts).
10. Any user-configured deny words or extra regular expressions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

KEY = "[REDACTED:key]"
EMAIL = "[REDACTED:email]"
NUMBER = "[REDACTED:number]"
ENV = "[REDACTED:env]"
CRED = "[REDACTED:cred]"
PRIVATE_KEY = "[REDACTED:private-key]"
CUSTOM = "[REDACTED:custom]"
PASTED_TAG = "[PASTED BLOCK]"


def pasted(lines: int) -> str:
    return f"[PASTED BLOCK: {lines} lines]"


_PEM = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)",
    re.DOTALL,
)
_PASTED_TAG = re.compile(
    r"<pasted[_-]content\b[^>]*>.*?(?:</pasted[_-]content>|\Z)", re.IGNORECASE | re.DOTALL
)
_FENCE = re.compile(r"(?ms)^[ \t]*(```|~~~)([^\n]*)\n(.*?)(?:^[ \t]*\1[ \t]*$|\Z)")

_SECRET_NAME = (
    r"[A-Za-z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PASS|PWD|AUTH|CREDENTIAL|PRIVATE|DSN|"
    r"SESSION|COOKIE|SALT|SIGNATURE|DATABASE_URL|CONN(?:ECTION)?_?STRING)[A-Za-z0-9_]*"
)
_ENV_ASSIGN_LINE = re.compile(r"(?m)^([ \t]*(?:export[ \t]+)?[A-Za-z_][A-Za-z0-9_]*[ \t]*=[ \t]*)(\S.*)$")
_VALUE = r"(\"[^\"\n]*\"|'[^'\n]*'|[^\s,;&`]+)"
_SECRET_ASSIGN = re.compile(
    r"(?<![A-Za-z0-9_])((?:export\s+)?" + _SECRET_NAME + r"\s*[:=]\s*)" + _VALUE
)
_SECRET_ASSIGN_LOWER = re.compile(
    r"(?i)(?<![A-Za-z0-9_])((?:api[_-]?key|secret|client[_-]?secret|password|passwd|"
    r"(?:access|auth|refresh|api)[_-]?token|token)\s*[:=]\s*)" + _VALUE
)
_JSON_SECRET = re.compile(
    r"(?i)(\"(?:" + _SECRET_NAME + r")\"\s*:\s*)\"[^\"\n]*\""
)
_URL_CRED = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@")
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:access_token|token|api_key|apikey|key|secret|password|pass|sig|signature|auth|code)=)"
    r"[^&\s#]+"
)
_KNOWN_TOKENS = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"sk-(?:ant-|proj-|live-|test-)?[A-Za-z0-9_-]{12,}"
    r"|gh[pousr]_[A-Za-z0-9]{16,}"
    r"|github_pat_[A-Za-z0-9_]{16,}"
    r"|xox[abprse]-[A-Za-z0-9-]{8,}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_-]{30,}"
    r"|glpat-[A-Za-z0-9_-]{16,}"
    r"|hf_[A-Za-z0-9]{20,}"
    r"|npm_[A-Za-z0-9]{20,}"
    r"|pypi-[A-Za-z0-9_-]{20,}"
    r"|(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{12,}"
    r"|whsec_[A-Za-z0-9]{16,}"
    r"|lin_api_[A-Za-z0-9]{16,}"
    r"|shp(?:at|ss|ca)_[A-Za-z0-9]{16,}"
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r")"
)
_AUTH_HEADER = re.compile(r"(?i)\b(bearer|basic|token)(\s+)[A-Za-z0-9._~+/=-]{12,}")
_GENERIC = re.compile(r"(?<![A-Za-z0-9+/_=-])[A-Za-z0-9+/_=-]{32,}(?![A-Za-z0-9+/_=-])")
_HEX = re.compile(r"^[0-9a-fA-F]+$")
_EMAIL = re.compile(r"(?i)(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_HOME_POSIX = re.compile(r"(?<![\w.-])/(?:Users|home)/[^/\s:'\"`]+")
_HOME_WIN = re.compile(r"(?i)[A-Z]:\\+Users\\+[^\\\s:'\"`]+")
_MONEY = re.compile(r"([$£€])\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:[kKmMbB]|bn|mm)\b)?")
_LONG_DIGITS = re.compile(r"(?<![\w.])\d(?:[ -]?\d){11,18}(?![\w.])")


@dataclass(frozen=True)
class RedactOptions:
    """Tunables for :func:`redact`. The defaults are what the pipeline uses."""

    max_block_lines: int = 12
    deny_words: tuple[str, ...] = ()
    extra_patterns: tuple[str, ...] = ()
    _compiled: tuple[re.Pattern[str], ...] = field(default=(), repr=False, compare=False)

    @classmethod
    def build(
        cls,
        max_block_lines: int = 12,
        deny_words: Iterable[str] = (),
        extra_patterns: Iterable[str] = (),
    ) -> "RedactOptions":
        words = tuple(w for w in deny_words if w.strip())
        extras = tuple(extra_patterns)
        compiled = tuple(re.compile(re.escape(w), re.IGNORECASE) for w in words) + tuple(
            re.compile(p) for p in extras
        )
        return cls(max_block_lines, words, extras, compiled)


DEFAULT_OPTIONS = RedactOptions()


def _fences(text: str, opts: RedactOptions) -> str:
    def repl(match: re.Match[str]) -> str:
        lang = match.group(2).strip().lower()
        body = match.group(3)
        lines = [ln for ln in body.splitlines() if ln.strip()]
        assigns = sum(1 for ln in lines if _ENV_ASSIGN_LINE.match(ln))
        if lang in {"env", "dotenv", ".env", "ini", "properties"} or ".env" in lang or (
            lines and assigns >= max(2, len(lines) // 2)
        ):
            return f"{ENV} block\n"
        if len(body.splitlines()) > opts.max_block_lines:
            return pasted(len(body.splitlines())) + "\n"
        return match.group(0)

    return _FENCE.sub(repl, text)


def _long_runs(text: str, opts: RedactOptions) -> str:
    """Collapse unfenced runs of many consecutive non-empty lines (pasted logs, code, files)."""
    lines = text.split("\n")
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= opts.max_block_lines:
            out.append(pasted(len(run)))
        else:
            out.extend(run)
        run.clear()

    for line in lines:
        if line.strip():
            run.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def _env(text: str) -> str:
    if ".env" in text.lower():
        text = _ENV_ASSIGN_LINE.sub(lambda m: m.group(1) + ENV, text)
    text = _JSON_SECRET.sub(lambda m: m.group(1) + '"' + ENV + '"', text)
    for pattern in (_SECRET_ASSIGN, _SECRET_ASSIGN_LOWER):
        text = pattern.sub(
            lambda m: m.group(0) if m.group(2).startswith(("[REDACTED", '"[REDACTED'))
            else m.group(1) + ENV,
            text,
        )
    return text


def _generic(match: re.Match[str]) -> str:
    token = match.group(0)
    core = token.strip("=")
    pieces = [p for p in re.split(r"[-_/.]", core) if p]
    if len(pieces) >= 3 and all(len(p) <= 16 for p in pieces):
        return token  # looks like a path, slug or branch name
    if _HEX.match(core) and len(core) >= 32:
        return KEY
    has_upper = any(c.isupper() for c in core)
    has_lower = any(c.islower() for c in core)
    has_digit = any(c.isdigit() for c in core)
    if has_upper and has_lower and has_digit:
        return KEY
    return token


def redact(text: str | None, opts: RedactOptions = DEFAULT_OPTIONS) -> str:
    """Return ``text`` with secrets and personal data replaced by fixed placeholders."""
    if not text:
        return ""
    value = str(text).replace("\r\n", "\n")
    value = _PEM.sub(PRIVATE_KEY, value)
    value = _PASTED_TAG.sub(PASTED_TAG, value)
    value = _fences(value, opts)
    value = _long_runs(value, opts)
    value = _URL_CRED.sub(lambda m: m.group(1) + CRED + "@", value)
    value = _QUERY_SECRET.sub(lambda m: m.group(1) + CRED, value)
    value = _env(value)
    value = _KNOWN_TOKENS.sub(KEY, value)
    value = _AUTH_HEADER.sub(lambda m: m.group(1) + m.group(2) + KEY, value)
    value = _GENERIC.sub(_generic, value)
    value = _EMAIL.sub(EMAIL, value)
    value = _HOME_POSIX.sub("~", value)
    value = _HOME_WIN.sub("~", value)
    value = _MONEY.sub(lambda m: m.group(1) + "N", value)
    value = _LONG_DIGITS.sub(NUMBER, value)
    for pattern in opts._compiled:
        value = pattern.sub(CUSTOM, value)
    return value


def short_quote(text: str, limit: int = 120, opts: RedactOptions = DEFAULT_OPTIONS) -> str:
    """A one-line, redacted, length-capped quote suitable for a report table."""
    one_line = " ".join(redact(text, opts).split())
    one_line = one_line.replace("|", "/")
    if len(one_line) > limit:
        one_line = one_line[: limit - 1].rstrip() + "\u2026"
    return one_line
