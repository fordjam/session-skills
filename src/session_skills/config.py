"""Configuration: an optional TOML file plus command-line overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .redact import RedactOptions

DEFAULT_CONFIG = "~/.config/session-skills/config.toml"


def default_state_dir() -> Path:
    if os.environ.get("SESSION_SKILLS_STATE"):
        return Path(os.environ["SESSION_SKILLS_STATE"]).expanduser()
    base = os.environ.get("XDG_STATE_HOME") or "~/.local/state"
    return Path(base).expanduser() / "session-skills"


@dataclass
class SourceConfig:
    enabled: bool = True
    paths: list[str] = field(default_factory=list)  # empty = adapter defaults


@dataclass
class Config:
    state_dir: Path = field(default_factory=default_state_dir)
    sources: dict[str, SourceConfig] = field(
        default_factory=lambda: {n: SourceConfig() for n in ("claude_code", "codex", "opencode")}
    )
    tag_backend: str = "claude:haiku"
    calibrate_backend: str = "codex:gpt-5-mini"
    max_requests: int = 200
    max_chars: int = 90_000
    retries: int = 3
    timeout: float = 300.0
    sample_size: int = 30
    min_agreement: float = 0.8
    default_threshold: float = 0.7
    min_chats: int = 3
    stale_days: int = 60
    repeated_max_chars: int = 200
    repeated_min_sessions: int = 3
    deny_words: list[str] = field(default_factory=list)
    extra_patterns: list[str] = field(default_factory=list)
    max_block_lines: int = 12

    def redact_options(self) -> RedactOptions:
        return RedactOptions.build(self.max_block_lines, self.deny_words, self.extra_patterns)


def load_config(path: str | None = None) -> Config:
    """Load ``path`` (or the default location if it exists) into a :class:`Config`."""
    cfg = Config()
    target = Path(path or DEFAULT_CONFIG).expanduser()
    if not target.is_file():
        if path:
            raise FileNotFoundError(f"config file not found: {path}")
        return cfg
    data = tomllib.loads(target.read_text(encoding="utf-8"))
    apply(cfg, data)
    return cfg


def apply(cfg: Config, data: dict[str, Any]) -> None:
    if "state_dir" in data:
        cfg.state_dir = Path(str(data["state_dir"])).expanduser()
    for name, section in (data.get("sources") or {}).items():
        if name not in cfg.sources:
            raise ValueError(f"unknown source in config: {name}")
        if isinstance(section, dict):
            cfg.sources[name] = SourceConfig(
                bool(section.get("enabled", True)), [str(p) for p in section.get("paths", [])]
            )
    tag = data.get("tag") or {}
    cfg.tag_backend = str(tag.get("backend", cfg.tag_backend))
    cfg.max_requests = int(tag.get("max_requests", cfg.max_requests))
    cfg.max_chars = int(tag.get("max_chars", cfg.max_chars))
    cfg.retries = int(tag.get("retries", cfg.retries))
    cfg.timeout = float(tag.get("timeout", cfg.timeout))
    cal = data.get("calibrate") or {}
    cfg.calibrate_backend = str(cal.get("backend", cfg.calibrate_backend))
    cfg.sample_size = int(cal.get("sample_size", cfg.sample_size))
    cfg.min_agreement = float(cal.get("min_agreement", cfg.min_agreement))
    cfg.default_threshold = float(cal.get("default_threshold", cfg.default_threshold))
    brief = data.get("brief") or {}
    cfg.min_chats = int(brief.get("min_chats", cfg.min_chats))
    cfg.stale_days = int(brief.get("stale_days", cfg.stale_days))
    rep = data.get("repeated") or {}
    cfg.repeated_max_chars = int(rep.get("max_chars", cfg.repeated_max_chars))
    cfg.repeated_min_sessions = int(rep.get("min_sessions", cfg.repeated_min_sessions))
    red = data.get("redact") or {}
    cfg.deny_words = [str(w) for w in red.get("deny_words", cfg.deny_words)]
    cfg.extra_patterns = [str(p) for p in red.get("extra_patterns", cfg.extra_patterns)]
    cfg.max_block_lines = int(red.get("max_block_lines", cfg.max_block_lines))
