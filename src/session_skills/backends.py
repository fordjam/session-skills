"""Worker backends: send one redacted prompt, get text back.

Spec strings (``--backend``):

  claude[:MODEL]              claude -p --model MODEL          (prompt on stdin, no tools, not persisted)
  codex[:MODEL]               codex exec -m MODEL -            (prompt on stdin, read-only, ephemeral)
  opencode:PROVIDER/MODEL     opencode run -m PROVIDER/MODEL   (prompt as an argument)
  ollama:MODEL                ollama run MODEL                 (prompt on stdin, fully local)
  exec:COMMAND                any command reading the prompt on stdin and printing the answer
  fake[:agree|noisy|broken]   deterministic offline heuristic, for tests and the demo

Every subprocess runs in a fresh empty temporary directory so an agentic CLI cannot touch the
current project. Model families are used by ``calibrate`` to insist on a *different* family for
the second opinion.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .questions import BY_NAME, QUESTION_NAMES


class BackendError(RuntimeError):
    pass


class Backend(Protocol):
    spec: str

    def family(self) -> str: ...

    def complete(self, prompt: str) -> str: ...


_FAMILY_HINTS = (
    ("claude", "anthropic"),
    ("anthropic", "anthropic"),
    ("gpt", "openai"),
    ("openai", "openai"),
    ("codex", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("gemini", "google"),
    ("gemma", "google"),
    ("google", "google"),
    ("llama", "meta"),
    ("qwen", "qwen"),
    ("deepseek", "deepseek"),
    ("mistral", "mistral"),
    ("mixtral", "mistral"),
    ("glm", "zhipu"),
    ("kimi", "moonshot"),
    ("grok", "xai"),
    ("phi", "microsoft"),
    ("mimo", "xiaomi"),
)


def family_of_model(model: str) -> str:
    lowered = model.lower()
    provider = lowered.split("/", 1)[0] if "/" in lowered else ""
    for hint, family in _FAMILY_HINTS:
        if provider and hint == provider:
            return family
    name = lowered.rsplit("/", 1)[-1]
    for hint, family in _FAMILY_HINTS:
        if re.search(rf"(^|[^a-z]){re.escape(hint)}", name):
            return family
    return f"unknown:{name or lowered}"


@dataclass
class CommandBackend:
    """Runs a CLI once per prompt."""

    spec: str
    argv: list[str]
    stdin: bool = True
    model_family: str = "unknown"
    timeout: float = 300.0
    last_message_flag: str | None = None  # e.g. codex --output-last-message
    env: dict[str, str] = field(default_factory=dict)

    def family(self) -> str:
        return self.model_family

    def complete(self, prompt: str) -> str:
        with tempfile.TemporaryDirectory(prefix="session-skills-") as work:
            argv = list(self.argv)
            last_file = Path(work) / "last-message.txt"
            if self.last_message_flag:  # goes before a trailing "-" (read prompt from stdin)
                insert_at = len(argv) - 1 if argv and argv[-1] == "-" else len(argv)
                argv[insert_at:insert_at] = [self.last_message_flag, str(last_file)]
            if not self.stdin:
                argv.append(prompt)
            try:
                done = subprocess.run(
                    argv,
                    input=prompt if self.stdin else None,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=work,
                    env={**os.environ, **self.env},
                    check=False,
                )
            except FileNotFoundError as exc:
                raise BackendError(f"{argv[0]} not found on PATH") from exc
            except subprocess.TimeoutExpired as exc:
                raise BackendError(f"{self.spec} timed out after {self.timeout:.0f}s") from exc
            if done.returncode != 0:
                tail = (done.stderr or "").strip().splitlines()[-1:] or [""]
                raise BackendError(f"{self.spec} exited {done.returncode}: {tail[0][:200]}")
            if self.last_message_flag and last_file.is_file():
                text = last_file.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    return text
            return done.stdout


@dataclass
class FakeBackend:
    """Deterministic, offline backend.

    With ``responder`` it returns whatever the callable returns. Otherwise it answers with a
    keyword heuristic over the human lines of the prompt, so the whole pipeline (including the
    demo) runs without any model. Variants: ``agree`` (default heuristic), ``noisy`` (flips a
    deterministic subset of answers, to exercise calibration) and ``broken`` (never valid JSON).
    """

    spec: str = "fake"
    variant: str = "agree"
    responder: Callable[[str], str] | None = None
    fake_family: str | None = None
    calls: list[str] = field(default_factory=list)

    def family(self) -> str:
        return self.fake_family or f"fake-{self.variant}"

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.responder is not None:
            return self.responder(prompt)
        if self.variant == "broken":
            return "I'm sorry, I can't produce JSON right now."
        human = " ".join(
            line[len("USER: ") :].lower()
            for line in prompt.splitlines()
            if line.startswith("USER: ")
        )
        answer: dict[str, object] = {}
        for name in QUESTION_NAMES:
            hit = any(cue in human for cue in BY_NAME[name].cues if len(cue) > 3)
            score = 0.9 if hit else 0.05
            if self.variant == "noisy":
                # systematic disagreement on two questions, rare flips elsewhere
                digest = hashlib.sha256(f"{name}:{human}".encode()).digest()[0]
                if name in {"tone", "format"} or digest < 4:
                    score = 1.0 - score
            answer[name] = score
        answer["task"] = "debugging" if "error" in human or "fail" in human else "coding"
        answer["outcome"] = "resolved" if "thanks" in human or "perfect" in human else "unclear"
        answer["frustration"] = 1 + min(4, sum(1 for n in QUESTION_NAMES if answer[n] == 0.9))
        return "```json\n" + json.dumps(answer) + "\n```"


def make_backend(spec: str, *, timeout: float = 300.0) -> Backend:
    kind, _, model = spec.partition(":")
    kind = kind.strip().lower()
    model = model.strip()
    if kind == "fake":
        return FakeBackend(spec=spec, variant=model or "agree")
    if kind == "claude":
        # no tools, and do not write this request into the session history we are mining
        argv = ["claude", "-p", "--output-format", "text", "--tools", "", "--no-session-persistence"]
        if model:
            argv += ["--model", model]
        return CommandBackend(spec, argv, True, "anthropic", timeout)
    if kind == "codex":
        argv = ["codex", "exec", "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral"]
        if model:
            argv += ["-m", model]
        argv.append("-")
        return CommandBackend(
            spec, argv, True, "openai", timeout, last_message_flag="--output-last-message"
        )
    if kind == "opencode":
        if not model:
            raise ValueError("opencode backend needs a model: opencode:PROVIDER/MODEL")
        return CommandBackend(spec, ["opencode", "run", "-m", model], False, family_of_model(model), timeout)
    if kind == "ollama":
        if not model:
            raise ValueError("ollama backend needs a model: ollama:MODEL")
        return CommandBackend(spec, ["ollama", "run", model], True, family_of_model(model), timeout)
    if kind == "exec":
        if not model:
            raise ValueError("exec backend needs a command: exec:COMMAND")
        return CommandBackend(spec, shlex.split(model), True, f"custom:{model}", timeout)
    raise ValueError(
        f"unknown backend {spec!r}; use claude[:MODEL], codex[:MODEL], opencode:MODEL, "
        "ollama:MODEL, exec:COMMAND or fake"
    )
