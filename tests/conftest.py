"""Shared fixtures. Every transcript here is synthetic; secret-shaped strings are assembled at
runtime so that no literal secret pattern appears in the repository."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from session_skills.config import Config, SourceConfig
from session_skills.state import State

TODAY = date(2026, 9, 29)


class Secrets:
    """Synthetic secret-shaped values, built by concatenation."""

    anthropic_key = "sk-" + "ant-" + "a1B2c3D4" * 4
    github_token = "ghp" + "_" + "Zz9" * 12
    aws_key = "AKIA" + "ABCDEFGHIJKLMNOP"
    slack_token = "xoxb" + "-" + "1234-5678-abcdEFGH"
    jwt = "eyJ" + "hbGciOiJIUzI1NiJ9" + "." + "eyJzdWIiOiIxMjM0NTY3ODkwIn0" + "." + "abcDEF123456ghiJKL"
    bearer_value = "abc123DEF456" + "ghi789JKL"
    generic = "Zx9Qw8Er7Ty6" + "Ui5Op4As3Df2Gh1Jk0Lz"
    hex40 = "0123456789abcdef" * 2 + "01234567"
    email = "jane.doe" + "@" + "corp.test"
    card = "4111" * 4
    card_spaced = " ".join(["4111"] * 4)
    account = "0012" + "3456" + "7890" + "12"
    home_user = "alice"
    home_mac = "/Users/" + "alice" + "/projects/app"
    home_linux = "/home/" + "bob" + "/src/tool"
    pem = "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIEow\n-----END " + "RSA PRIVATE KEY-----"
    password = "hunter2-" + "sup3r"
    session_cookie = "abc123" + "abc123"
    putty = "PuTTY-User-Key-File-" + "3: ssh-ed25519\nPrivate-Lines: 1\nAAAAsecretbody\nPrivate-MAC: 00ff"
    pgp = "-----BEGIN PGP " + "PRIVATE KEY BLOCK-----\nlQOYBF\n-----END PGP " + "PRIVATE KEY BLOCK-----"
    jwt_unsigned = "eyJ" + "hbGciOiJub25lIn0" + "." + "eyJzdWIiOiIxMjM0In0" + "."

    @classmethod
    def all_values(cls) -> list[str]:
        return [
            cls.anthropic_key, cls.github_token, cls.aws_key, cls.slack_token, cls.jwt,
            cls.generic, cls.hex40, cls.email, cls.card, cls.account,
        ]


@pytest.fixture
def secrets() -> type[Secrets]:
    return Secrets


@pytest.fixture
def roots(tmp_path: Path) -> dict[str, Path]:
    return {
        "claude_code": tmp_path / "claude" / "projects",
        "codex": tmp_path / "codex",
        "opencode": tmp_path / "opencode",
    }


@pytest.fixture
def cfg(tmp_path: Path, roots: dict[str, Path]) -> Config:
    config = Config(state_dir=tmp_path / "state")
    config.retry_delay = 0
    config.sources = {
        "claude_code": SourceConfig(True, [str(roots["claude_code"] / "**" / "*.jsonl")]),
        "codex": SourceConfig(True, [str(roots["codex"] / "sessions" / "**" / "*.jsonl")]),
        "opencode": SourceConfig(True, [str(roots["opencode"] / "*.db"), str(roots["opencode"] / "storage")]),
    }
    return config


@pytest.fixture
def state(cfg: Config) -> Iterator[State]:
    opened = State(cfg.state_dir)
    yield opened
    opened.close()
