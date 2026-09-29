"""Opt-in checks against the real agent CLIs (no model is called, no session is read).

Run with ``SESSION_SKILLS_REAL_CLI=1 pytest tests/test_real_cli.py``. They ask each installed CLI
how it resolved the isolation settings this tool passes, instead of trusting argument strings.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from session_skills.backends import CODEX_DISABLED_FEATURES, OPENCODE_AGENT, CommandBackend, make_backend

pytestmark = pytest.mark.skipif(os.environ.get("SESSION_SKILLS_REAL_CLI") != "1", reason="set SESSION_SKILLS_REAL_CLI=1")


@pytest.mark.skipif(shutil.which("opencode") is None, reason="opencode not installed")
def test_opencode_resolves_every_tool_off_and_denied(tmp_path: Path):
    backend = make_backend("opencode:anthropic/claude-haiku-4-5")
    assert isinstance(backend, CommandBackend)
    for rel, content in backend.config_files.items():
        (tmp_path / rel).write_text(content.replace("{work}", str(tmp_path)))
    (tmp_path / "cwd").mkdir()
    env = backend.environment(str(tmp_path))
    done = subprocess.run(["opencode", "--pure", "debug", "agent", OPENCODE_AGENT], cwd=tmp_path / "cwd", env=env,
                          capture_output=True, text=True, timeout=120, check=True)
    agent = json.loads(done.stdout)
    assert not any(agent["tools"].values())
    wildcard = [r["action"] for r in agent["permission"] if r["permission"] == "*" and r["pattern"] == "*"]
    assert wildcard[-1] == "deny"
    for tool in ("bash", "edit", "read", "webfetch", "task", "skill"):
        rules = [r["action"] for r in agent["permission"] if r["permission"] in (tool, "*") and r["pattern"] == "*"]
        assert rules[-1] == "deny", tool
    assert (tmp_path / "data").exists()  # sessions would land in the throwaway data dir


@pytest.mark.skipif(shutil.which("codex") is None, reason="codex not installed")
def test_codex_reports_the_disabled_features_off(tmp_path: Path):
    args = [a for f in CODEX_DISABLED_FEATURES for a in ("--disable", f)]
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "CODEX_HOME": str(tmp_path / "codex")}
    (tmp_path / "codex").mkdir()
    done = subprocess.run(["codex", *args, "features", "list"], env=env, capture_output=True, text=True,
                          timeout=60, check=True)
    state = {line.split()[0]: line.split()[-1] for line in done.stdout.splitlines() if line.strip()}
    for feature in CODEX_DISABLED_FEATURES:
        assert state.get(feature) == "false", feature
