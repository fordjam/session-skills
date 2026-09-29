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
from session_skills.backends import (
    BASE_ENV,
    BackendError,
    CommandBackend,
    FatalBackendError,
    family_of_model,
    family_of_spec,
    make_backend,
)
from session_skills.config import Config, load_config
from session_skills.install import InstallError, install, rollback
from session_skills.questions import QUESTION_NAMES
from session_skills.synthetic import Conversation, write_claude_code

ANSWER = {**dict.fromkeys(QUESTION_NAMES, 0.1), "too_long": 0.95, "task": "coding", "outcome": "resolved", "frustration": 3}
CANARY = "SESSION_SKILLS_TEST_CANARY"


def fake_cli(bin_dir: Path, name: str, body: str, log: Path) -> Path:
    """Write an executable fake CLI that records its argv, cwd, environment and generated config."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    path = bin_dir / name
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"log = {str(log)!r}\n"
        "prompt = sys.stdin.read() if not sys.stdin.isatty() else ''\n"
        "config = None\n"
        "if os.environ.get('OPENCODE_CONFIG'):\n"
        "    config = json.load(open(os.environ['OPENCODE_CONFIG']))\n"
        "auth = os.path.join(os.environ.get('XDG_DATA_HOME', '/nonexistent'), 'opencode', 'auth.json')\n"
        "with open(log, 'a') as fh: fh.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'stdin': prompt,"
        " 'env': dict(os.environ), 'config': config, 'cwd_files': os.listdir('.'), 'auth_link': os.path.islink(auth)}) + '\\n')\n"
        + body
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def fake_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv(CANARY, "must-not-reach-a-worker")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "must-not-reach-a-worker-either")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
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
    assert family_of_spec("claude:sonnet") == family_of_spec("claude:haiku") == "anthropic"
    with pytest.raises(ValueError):
        make_backend("opencode")
    with pytest.raises(ValueError):
        make_backend("nope:x")


# -- isolation: no backend may get write or shell capability -------------------------------------

FORBIDDEN_FLAGS = {
    "--dangerously-skip-permissions", "--allow-dangerously-skip-permissions", "--allowed-tools", "--allowedTools",
    "--add-dir", "--mcp-config", "--plugin-dir", "--plugin-url", "--agents", "--chrome", "--auto",
    "--dangerously-bypass-approvals-and-sandbox", "--full-auto", "--approve-for-me", "--worktree", "--share",
    "--continue", "--resume", "--session",
}  # fmt: skip
SPECS = ["claude:haiku", "codex:some-model", "opencode:anthropic/claude-haiku", "ollama:llama3.2", "exec:worker"]


@pytest.mark.parametrize("spec", SPECS)
def test_no_backend_gets_write_or_shell_capability(spec, fake_path, tmp_path, monkeypatch):
    bin_dir, log = fake_path
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    (tmp_path / "share" / "opencode").mkdir(parents=True)
    (tmp_path / "share" / "opencode" / "auth.json").write_text("{}")
    binary = spec.split(":")[0] if not spec.startswith("exec:") else "worker"
    fake_cli(bin_dir, binary, "i = sys.argv.index('--output-last-message') if '--output-last-message' in sys.argv else None\n"
             f"out = {json.dumps(json.dumps(ANSWER))}\n"
             "open(sys.argv[i + 1], 'w').write(out) if i is not None else print(out)\n", log)
    backend = make_backend(spec)
    backend.complete("the prompt")
    call = calls(log)[0]
    argv, env = call["argv"], call["env"]

    # nothing from your environment beyond the allowlist; nothing secret-looking by default
    assert CANARY not in env and "SOME_SERVICE_TOKEN" not in env
    allowed = set(BASE_ENV) | {"TERM", "NO_COLOR"}
    extra = {k for k in env if k not in allowed}
    assert all(k.startswith(("CLAUDE", "DISABLE_", "OPENCODE_", "XDG_", "OLLAMA_", "CODEX_", "ANTHROPIC_", "OPENAI_"))
               or k.endswith("_API_KEY") or k.startswith("__CF") for k in extra), extra
    # a fresh empty working directory, never the project
    assert Path(call["cwd"]).name == "cwd" and call["cwd_files"] == []
    assert not Path(call["cwd"]).exists()
    # no flag that grants tools, approvals, extra directories or session reuse
    assert not FORBIDDEN_FLAGS & set(argv)
    info = backend.describe()
    assert info["tools"] in {"none", "read-only-sandbox"}

    if spec.startswith("claude"):
        assert argv[argv.index("--tools") + 1] == ""
        assert {"--safe-mode", "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"} <= set(argv)
        assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
        assert call["stdin"] == "the prompt"
    elif spec.startswith("codex"):
        assert argv[argv.index("--sandbox") + 1] == "read-only"
        assert {"--ephemeral", "--ignore-user-config", "--ignore-rules"} <= set(argv)
        assert "workspace-write" not in argv and "danger-full-access" not in argv
        disabled = {argv[i + 1] for i, a in enumerate(argv) if a == "--disable"}
        assert "shell_tool" in disabled
        assert 'approval_policy="never"' in argv and 'shell_environment_policy.inherit="none"' in argv
        assert any("residual" in note for note in info["notes"])  # documented, not hidden
    elif spec.startswith("opencode"):
        config = call["config"]
        assert config["permission"] == "deny" and config["mcp"] == {} and config["plugin"] == []
        agent = config["agent"][config["default_agent"]]
        assert agent["permission"] == "deny" and not any(agent["tools"].values()) and not any(config["tools"].values())
        assert argv[:4] == ["run", "--pure", "--agent", config["default_agent"]]
        work = os.path.realpath(Path(call["cwd"]).parent)
        assert os.path.realpath(env["XDG_DATA_HOME"]).startswith(work)  # throwaway session store
        assert os.path.realpath(env["XDG_CONFIG_HOME"]).startswith(work)  # your config is not loaded
        assert env["OPENCODE_DISABLE_CLAUDE_CODE"] == "1" and call["auth_link"]
        assert not Path(env["XDG_DATA_HOME"]).exists()
    elif spec.startswith("ollama"):
        assert argv == ["run", "--nowordwrap", "llama3.2"] and env["OLLAMA_HOST"] == "http://127.0.0.1:11434"
        assert info["destination"] == "local"


def test_codex_backend_reads_last_message_file(fake_path):
    bin_dir, log = fake_path
    fake_cli(
        bin_dir,
        "codex",
        "i = sys.argv.index('--output-last-message')\n"
        f"open(sys.argv[i + 1], 'w').write({json.dumps(json.dumps(ANSWER))})\n"
        "print('progress noise, not json')\n",
        log,
    )
    out = make_backend("codex:some-model").complete("p")
    assert json.loads(out)["frustration"] == 3
    argv = calls(log)[0]["argv"]
    assert argv[0] == "exec" and argv[-1] == "-" and "some-model" in argv


def test_pass_env_is_explicit(fake_path, monkeypatch):
    bin_dir, log = fake_path
    fake_cli(bin_dir, "worker", "print('{}')\n", log)
    make_backend("exec:worker", pass_env=[CANARY]).complete("x")
    assert calls(log)[0]["env"][CANARY] == "must-not-reach-a-worker"


def test_command_backend_errors(fake_path):
    bin_dir, log = fake_path
    fake_cli(bin_dir, "failing", "sys.stderr.write('boom\\n'); sys.exit(3)\n", log)
    with pytest.raises(BackendError, match="exited 3"):
        CommandBackend("exec:failing", ["failing"]).complete("x")
    with pytest.raises(FatalBackendError, match="not found"):
        CommandBackend("exec:missing", ["definitely-not-a-real-binary-xyz"]).complete("x")


# -- ollama is local only unless --allow-remote ---------------------------------------------------

@pytest.mark.parametrize("host", ["", "127.0.0.1:11434", "http://localhost:11434", "[::1]:11434", "0.0.0.0"])
def test_ollama_local_hosts_are_accepted(host, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", host)
    backend = make_backend("ollama:qwen3:8b", local_only=True)
    assert backend.describe()["destination"] == "local"


@pytest.mark.parametrize(("host", "model"), [
    ("https://ollama.example.org", "llama3.2"),
    ("10.0.0.5:11434", "llama3.2"),
    ("", "gpt-oss:120b-cloud"),
    ("", "gemma4:cloud"),
    ("", "deepseek-v3.1:671b-cloud"),
])
def test_ollama_remote_or_cloud_is_refused(host, model, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", host)
    with pytest.raises(FatalBackendError, match="off this machine"):
        make_backend(f"ollama:{model}")
    assert make_backend(f"ollama:{model}", allow_remote=True).describe()["destination"] == "remote"
    with pytest.raises(FatalBackendError):  # local-only wins over allow-remote
        make_backend(f"ollama:{model}", allow_remote=True, local_only=True)


@pytest.mark.parametrize("spec", ["claude:haiku", "codex", "opencode:ollama/llama3.2", "exec:worker"])
def test_local_only_refuses_everything_that_is_not_provably_local(spec):
    with pytest.raises(FatalBackendError, match="local-only"):
        make_backend(spec, local_only=True)


# -- CLI end to end ----------------------------------------------------------------------------

def write_config(tmp_path: Path, roots: dict[str, Path], backend: str = "fake") -> Path:
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
backend = "{backend}"
retry_delay = 0
[calibrate]
backend = "fake:noisy"
sample_size = 30
[redact]
deny_words = ["Initech"]
"""
    )
    return config


def seed(roots: dict[str, Path], n: int = 4) -> None:
    for i in range(n):
        write_claude_code(roots["claude_code"], Conversation(f"s{i}", TODAY - timedelta(days=i), [
            ("user", "status?"), ("assistant", "working"), ("user", "too long, shorter please, Initech wants it short")]))


def test_cli_pipeline(tmp_path, roots, capsys, monkeypatch):
    seed(roots)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-home"))
    config = str(write_config(tmp_path, roots))
    assert cli.run(["--config", config, "build-list"]) == 0
    assert "claude_code 4" in capsys.readouterr().out
    assert cli.run(["--config", config, "tag", "--dry-run"]) == 0
    assert "Would send 4" in capsys.readouterr().out
    assert cli.run(["--config", config, "--json", "tag"]) == 0
    assert json.loads(capsys.readouterr().out)["tagged"] == 4
    assert cli.run(["--config", config, "calibrate"]) == 0
    assert "Labelled 4/4" in capsys.readouterr().out
    assert cli.run(["--config", config, "repeated"]) == 0
    assert "status" in capsys.readouterr().out
    assert cli.run(["--config", config, "brief", "--today", TODAY.isoformat()]) == 0
    assert "keep-replies-short" in capsys.readouterr().out
    report = (tmp_path / "state" / "out" / "report.md").read_text()
    assert "Initech" not in report and "[REDACTED:custom]" in report

    # install goes to $CLAUDE_CONFIG_DIR/skills by default, after a diff and a confirmation
    target = tmp_path / "claude-home" / "skills"
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # not a tty: must refuse without --yes
    assert cli.run(["--config", config, "install", "--all"]) == 2
    assert "+name: keep-replies-short" in capsys.readouterr().out  # the diff was shown
    assert not target.exists()
    assert cli.run(["--config", config, "install", "keep-replies-short", "--yes"]) == 0
    assert (target / "keep-replies-short" / "SKILL.md").is_file()
    assert "rollback" in capsys.readouterr().out
    assert cli.run(["--config", config, "rollback"]) == 0
    assert not (target / "keep-replies-short").exists()


def test_cli_tag_failures_exit_nonzero_with_reasons(tmp_path, roots, capsys):
    seed(roots, 2)
    config = str(write_config(tmp_path, roots, backend="fake:broken"))
    assert cli.run(["--config", config, "build-list"]) == 0
    code = cli.run(["--config", config, "tag"])
    out = capsys.readouterr().out
    assert code == 1
    assert "failed 2" in out and "failure x2: no JSON object with question keys" in out
    assert cli.run(["--config", config, "--json", "tag"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["failures"] and data["errors"] == {"no JSON object with question keys": 2}


def test_cli_backend_that_cannot_run_exits_3(tmp_path, roots, capsys):
    seed(roots, 1)
    config = str(write_config(tmp_path, roots, backend="exec:definitely-not-a-real-binary-xyz"))
    assert cli.run(["--config", config, "build-list"]) == 0
    assert cli.run(["--config", config, "tag"]) == 3
    assert "not found" in capsys.readouterr().out


def test_cli_preview_shows_exact_command_and_payload(tmp_path, roots, capsys, secrets):
    write_claude_code(roots["claude_code"], Conversation("p", TODAY, [("user", f"too long; key {secrets.github_token}")]))
    config = str(write_config(tmp_path, roots, backend="claude:haiku"))
    assert cli.run(["--config", config, "build-list"]) == 0
    capsys.readouterr()
    assert cli.run(["--config", config, "tag", "--preview"]) == 0
    out = capsys.readouterr().out
    assert "--tools" in out and "--no-session-persistence" in out and "exact payload" in out
    assert secrets.github_token not in out and "[REDACTED:key]" in out
    assert "PATH" in out  # environment names, never values


def test_install_confirmation_and_errors(cfg, state, tmp_path):
    with pytest.raises(InstallError, match="brief"):
        install(state, [], all_skills=True, target=str(tmp_path / "t"), confirm=lambda p: True)
    skill = state.root / "out" / "skills" / "demo-skill" / "SKILL.md"
    state.write_text(skill, "---\nname: demo-skill\ndescription: x\n---\n")
    with pytest.raises(InstallError, match="cancelled"):
        install(state, ["demo-skill"], target=str(tmp_path / "t"), confirm=lambda p: False)
    assert not (tmp_path / "t").exists()
    with pytest.raises(InstallError, match="unknown"):
        install(state, ["other"], target=str(tmp_path / "t"), confirm=lambda p: True)
    plan, backup = install(state, ["demo-skill"], target=str(tmp_path / "t"), confirm=lambda p: True)
    assert plan.copy == ["demo-skill"] and (tmp_path / "t" / "demo-skill" / "SKILL.md").is_file() and backup


@pytest.mark.parametrize("agent", ["claude", "opencode", "codex"])
def test_agent_aware_install_with_rules_diff_and_rollback(agent, cfg, state, tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-cfg"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    expected = {
        "claude": (tmp_path / "claude-cfg" / "skills", tmp_path / "claude-cfg" / "CLAUDE.md"),
        "opencode": (tmp_path / "xdg" / "opencode" / "skills", tmp_path / "xdg" / "opencode" / "AGENTS.md"),
        "codex": (home / ".agents" / "skills", tmp_path / "codex-home" / "AGENTS.md"),
    }[agent]
    out = state.root / "out"
    state.write_text(out / "skills" / "keep-replies-short" / "SKILL.md", "---\nname: keep-replies-short\n---\nbody\n")
    state.write_text(out / "AGENTS.md", "<!-- session-skills:begin -->\n## Working with this user\n- rule one\n<!-- session-skills:end -->\n")
    state.write_text(out / "CLAUDE.md.snippet", "## Working with me\n- rule one\n")
    skills_dir, rules_file = expected
    rules_file.parent.mkdir(parents=True)
    original = "# My own instructions\n\nKeep this line.\n"
    rules_file.write_text(original)

    seen = {}

    def confirm(plan):
        seen["diffs"] = [c.diff() for c in plan.changes]
        return True

    install(state, ["keep-replies-short"], agent=agent, rules=True, confirm=confirm)
    assert (skills_dir / "keep-replies-short" / "SKILL.md").read_text().endswith("body\n")
    merged = rules_file.read_text()
    assert merged.startswith(original.rstrip()) and "- rule one" in merged and "session-skills:end" in merged
    assert any("+- rule one" in d for d in seen["diffs"])
    # a second install replaces only the block
    state.write_text(out / "AGENTS.md", "<!-- session-skills:begin -->\n- rule two\n<!-- session-skills:end -->\n")
    state.write_text(out / "CLAUDE.md.snippet", "- rule two\n")
    install(state, [], agent=agent, rules=True, confirm=lambda p: True)
    again = rules_file.read_text()
    assert "rule two" in again and "rule one" not in again and again.count("session-skills:begin") == 1
    assert "Keep this line." in again
    # rolling back twice restores the original bytes and removes the created skill
    rollback(state)
    rollback(state)
    assert rules_file.read_text() == original
    assert not (skills_dir / "keep-replies-short").exists()


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


# -- documentation matches the code ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_matches_code_defaults():
    example = load_config(str(ROOT / "examples" / "config.toml"))
    defaults = Config()
    for name in ("tag_backend", "calibrate_backend", "max_requests", "max_chars", "retries", "timeout", "sample_size",
                 "calibrate_max_requests", "min_agreement", "min_support", "min_positives", "default_threshold",
                 "min_chats", "stale_days", "repeated_max_chars", "repeated_min_sessions", "max_block_lines",
                 "local_only", "allow_remote", "retry_delay"):
        assert getattr(example, name) == getattr(defaults, name), name


def test_readme_is_consistent_with_code():
    readme = (ROOT / "README.md").read_text()
    assert "<you>" not in readme and "github.com/OWNER/session-skills" in readme
    defaults = Config()
    assert f"`{defaults.tag_backend}` (default tagger)" in readme
    assert f"`{defaults.calibrate_backend}` (default calibrator)" in readme
    assert f"at most {defaults.max_requests} backend calls per run" in readme
    assert "keeps everything on your machine" not in readme  # the old, unconditional claim


# -- round 2: ollama locality is checked by IP and by the server's own model metadata -----------

@pytest.mark.parametrize("host", [
    "https://127.attacker.example:11434",  # round-2 reviewer's bypass
    "127.0.0.1.evil.example",
    "localhost.evil.example",
    "http://[::ffff:10.0.0.1]:11434",
    "ftp://127.0.0.1",
    "not a host at all:::",
])
def test_ollama_hostnames_are_not_trusted_by_prefix(host, monkeypatch, local_ollama):
    monkeypatch.setenv("OLLAMA_HOST", host)
    with pytest.raises(FatalBackendError):
        make_backend("ollama:llama3.2", local_only=True)
    assert local_ollama == []  # nothing was contacted


@pytest.mark.parametrize(("host", "base"), [
    ("127.0.0.2:8080", "http://127.0.0.2:8080"),
    ("http://[::1]:11434", "http://[::1]:11434"),
    ("", "http://127.0.0.1:11434"),
])
def test_ollama_verifies_the_model_on_the_loopback_server(host, base, monkeypatch, local_ollama):
    monkeypatch.setenv("OLLAMA_HOST", host)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    backend = make_backend("ollama:llama3.2", local_only=True)
    assert local_ollama == [(base, "llama3.2")]
    assert backend.describe()["destination"] == "local"
    assert "HTTPS_PROXY" not in backend.describe()["env"]  # a proxy could forward "local" traffic


def test_ollama_model_that_is_missing_or_remote_is_refused(monkeypatch):
    import urllib.error

    from session_skills import backends

    def missing(base_url, model, timeout=10.0):
        raise urllib.error.HTTPError(base_url, 404, "not found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(backends, "_ollama_show", missing)
    with pytest.raises(FatalBackendError, match="not installed"):
        make_backend("ollama:llama3.2")

    monkeypatch.setattr(backends, "_ollama_show", lambda *a, **k: {"remote_model": "big", "remote_host": "https://ollama.com:443"})
    with pytest.raises(FatalBackendError, match="runs remotely"):
        make_backend("ollama:my-alias")  # a local-looking name that is really a cloud model

    def down(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(backends, "_ollama_show", down)
    with pytest.raises(FatalBackendError, match="could not reach"):
        make_backend("ollama:llama3.2")
    assert make_backend("ollama:llama3.2", allow_remote=True).describe()["destination"] == "remote"


# -- round 3: the endpoint that is verified is the endpoint that is used ------------------------

@pytest.mark.parametrize(("raw", "canonical"), [
    ("", "http://127.0.0.1:11434"),
    ("127.0.0.1", "http://127.0.0.1:11434"),
    ("127.0.0.1:8080", "http://127.0.0.1:8080"),
    ("http://127.0.0.1", "http://127.0.0.1:80"),  # round-3 reviewer: Ollama uses port 80 here
    ("https://localhost", "https://localhost:443"),
    ("http://[::1]:11434/", "http://[::1]:11434"),
])
def test_ollama_checks_and_runs_the_same_endpoint(raw, canonical, fake_path, monkeypatch, local_ollama):
    bin_dir, log = fake_path
    fake_cli(bin_dir, "ollama", f"print({json.dumps(json.dumps(ANSWER))})\n", log)
    monkeypatch.setenv("OLLAMA_HOST", raw)
    backend = make_backend("ollama:llama3.2", local_only=True)
    monkeypatch.setenv("OLLAMA_HOST", "http://10.9.9.9:11434")  # changing it afterwards changes nothing
    backend.complete("p")
    checked = [base for base, _ in local_ollama]
    used = calls(log)[0]["env"]["OLLAMA_HOST"]
    assert checked == [canonical] and used == canonical


@pytest.mark.parametrize("raw", [
    "http://127.0.0.1:11434/some/prefix",  # round-3 reviewer: paths were dropped for the check only
    "http://127.0.0.1:11434/?x=1",
    "http://user:pw@127.0.0.1:11434",
    "tcp://127.0.0.1:11434",
])
def test_ollama_hosts_we_cannot_canonicalise_are_refused(raw, monkeypatch, local_ollama):
    monkeypatch.setenv("OLLAMA_HOST", raw)
    with pytest.raises(FatalBackendError, match="scheme://host:port"):
        make_backend("ollama:llama3.2")
    assert local_ollama == []
