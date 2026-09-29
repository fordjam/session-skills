from __future__ import annotations

import io
import json
import os
import stat
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from conftest import TODAY
from session_skills import cli
from session_skills.backends import BackendError, CommandBackend, family_of_model, make_backend
from session_skills.install import InstallError, install
from session_skills.questions import QUESTION_NAMES
from session_skills.synthetic import Conversation, write_claude_code

ANSWER = {**{n: 0.1 for n in QUESTION_NAMES}, "too_long": 0.95, "task": "coding", "outcome": "resolved", "frustration": 3}


def fake_cli(bin_dir: Path, name: str, body: str) -> Path:
    """Write an executable fake CLI that records its argv and cwd."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    path = bin_dir / name
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "log = os.environ['FAKE_LOG']\n"
        "prompt = sys.stdin.read() if not sys.stdin.isatty() else ''\n"
        "with open(log, 'a') as fh: fh.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'stdin': len(prompt)}) + '\\n')\n"
        + body
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def fake_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LOG", str(log))
    return bin_dir, log


def calls(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_backend_specs_and_families():
    assert make_backend("claude:haiku").family() == "anthropic"
    assert make_backend("codex").family() == "openai"
    assert make_backend("opencode:openrouter/qwen/qwen3-coder").family() == "qwen"
    assert make_backend("ollama:llama3.2").family() == "meta"
    assert make_backend("fake:noisy").family() == "fake-noisy"
    assert family_of_model("anthropic/claude-haiku") == "anthropic"
    assert family_of_model("some-new-model").startswith("unknown:")
    with pytest.raises(ValueError):
        make_backend("opencode")
    with pytest.raises(ValueError):
        make_backend("nope:x")


def test_claude_backend_runs_outside_the_project(fake_path, tmp_path):
    bin_dir, log = fake_path
    fake_cli(bin_dir, "claude", f"print({json.dumps(json.dumps(ANSWER))})\n")
    backend = make_backend("claude:haiku")
    out = backend.complete("hello prompt")
    assert json.loads(out)["too_long"] == 0.95
    call = calls(log)[0]
    assert call["argv"][:1] == ["-p"] and "--model" in call["argv"] and "haiku" in call["argv"]
    assert "--no-session-persistence" in call["argv"] and call["argv"][call["argv"].index("--tools") + 1] == ""
    assert call["stdin"] == len("hello prompt")
    assert Path(call["cwd"]).name.startswith("session-skills-") and not Path(call["cwd"]).exists()


def test_codex_backend_reads_last_message_file(fake_path):
    bin_dir, log = fake_path
    fake_cli(
        bin_dir,
        "codex",
        "i = sys.argv.index('--output-last-message')\n"
        f"open(sys.argv[i + 1], 'w').write({json.dumps(json.dumps(ANSWER))})\n"
        "print('progress noise, not json')\n",
    )
    out = make_backend("codex:some-model").complete("p")
    assert json.loads(out)["frustration"] == 3
    argv = calls(log)[0]["argv"]
    assert argv[0] == "exec" and argv[-1] == "-" and "read-only" in argv and "some-model" in argv
    assert "--ephemeral" in argv


def test_opencode_backend_passes_prompt_as_argument(fake_path):
    bin_dir, log = fake_path
    fake_cli(bin_dir, "opencode", "print('{}')\n")
    make_backend("opencode:anthropic/claude-haiku").complete("the prompt")
    argv = calls(log)[0]["argv"]
    assert argv == ["run", "-m", "anthropic/claude-haiku", "the prompt"]


def test_command_backend_errors(fake_path):
    bin_dir, _ = fake_path
    fake_cli(bin_dir, "failing", "sys.stderr.write('boom\\n'); sys.exit(3)\n")
    with pytest.raises(BackendError, match="exited 3"):
        CommandBackend("exec:failing", ["failing"]).complete("x")
    with pytest.raises(BackendError, match="not found"):
        CommandBackend("exec:missing", ["definitely-not-a-real-binary-xyz"]).complete("x")


# -- CLI end to end ----------------------------------------------------------------------------

def write_config(tmp_path: Path, roots: dict[str, Path]) -> Path:
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
state_dir = "{tmp_path / 'state'}"
[sources.claude_code]
paths = ["{roots['claude_code']}/**/*.jsonl"]
[sources.codex]
enabled = false
[sources.opencode]
enabled = false
[tag]
backend = "fake"
[calibrate]
backend = "fake:noisy"
sample_size = 30
[redact]
deny_words = ["Initech"]
"""
    )
    return config


def test_cli_pipeline(tmp_path, roots, capsys, monkeypatch, fake_path):
    bin_dir, _ = fake_path
    fake_cli(bin_dir, "labeller", f"print({json.dumps(json.dumps(ANSWER))})\n")
    for i in range(4):
        write_claude_code(roots["claude_code"], Conversation(f"s{i}", TODAY - timedelta(days=i), [
            ("user", "status?"), ("assistant", "working"), ("user", "too long, shorter please, Initech wants it short")]))
    config = str(write_config(tmp_path, roots))
    assert cli.run(["--config", config, "build-list"]) == 0
    assert "claude_code 4" in capsys.readouterr().out
    assert cli.run(["--config", config, "tag", "--dry-run"]) == 0
    assert "Would send 4" in capsys.readouterr().out
    assert cli.run(["--config", config, "--json", "tag"]) == 0
    assert json.loads(capsys.readouterr().out)["tagged"] == 4
    assert cli.run(["--config", config, "calibrate", "--backend", "exec:labeller"]) == 0
    assert "Labelled 4/4" in capsys.readouterr().out
    assert cli.run(["--config", config, "repeated"]) == 0
    assert "status" in capsys.readouterr().out
    assert cli.run(["--config", config, "brief", "--today", TODAY.isoformat()]) == 0
    out = capsys.readouterr().out
    assert "keep-replies-short" in out
    report = (tmp_path / "state" / "out" / "report.md").read_text()
    assert "Initech" not in report and "[REDACTED:custom]" in report

    target = tmp_path / "skills-target"
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # not a tty: must refuse without --yes
    assert cli.run(["--config", config, "install", "--all", "--target", str(target)]) == 2
    assert not target.exists()
    assert cli.run(["--config", config, "install", "keep-replies-short", "--target", str(target), "--yes"]) == 0
    assert (target / "keep-replies-short" / "SKILL.md").is_file()
    assert cli.run(["--config", config, "install", "keep-replies-short", "--target", str(target), "--yes"]) == 0
    assert "Skipped" in capsys.readouterr().out


def test_install_confirmation_and_errors(cfg, state, roots, tmp_path):
    with pytest.raises(InstallError, match="brief"):
        install(state, [], all_skills=True, target=str(tmp_path / "t"), confirm=lambda p: True)
    skill = state.root / "out" / "skills" / "demo-skill" / "SKILL.md"
    state.write_text(skill, "---\nname: demo-skill\ndescription: x\n---\n")
    with pytest.raises(InstallError, match="cancelled"):
        install(state, ["demo-skill"], target=str(tmp_path / "t"), confirm=lambda p: False)
    assert not (tmp_path / "t").exists()
    with pytest.raises(InstallError, match="unknown"):
        install(state, ["other"], target=str(tmp_path / "t"), confirm=lambda p: True)
    plan = install(state, ["demo-skill"], target=str(tmp_path / "t"), confirm=lambda p: True)
    assert plan.copy == ["demo-skill"] and (tmp_path / "t" / "demo-skill" / "SKILL.md").is_file()


def test_cli_redact_and_demo(tmp_path, capsys, monkeypatch, secrets):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--config", str(tmp_path / "missing.toml"), "redact"])
    assert exc.value.code == 2
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"key {secrets.github_token}\n"))
    monkeypatch.setenv("HOME", str(tmp_path))  # no default config file
    assert cli.run(["redact"]) == 0
    assert capsys.readouterr().out == "key [REDACTED:key]\n"
    assert cli.run(["--json", "demo", "--dir", str(tmp_path / "demo"), "--today", TODAY.isoformat()]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "too_long" in result["patterns"] and "jargon" in result["stale"]
    assert result["automation_candidates"] >= 3


def test_state_dir_inside_repo_is_rejected(tmp_path, capsys):
    (tmp_path / ".git").mkdir()
    with pytest.raises(SystemExit) as exc:
        cli.main(["--state-dir", str(tmp_path / "state"), "repeated"])
    assert exc.value.code == 2
    assert "inside a repository" in capsys.readouterr().err
