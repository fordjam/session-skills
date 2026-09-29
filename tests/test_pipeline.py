from __future__ import annotations

import json
import stat
from datetime import timedelta
from pathlib import Path

import pytest

from conftest import TODAY
from session_skills.backends import FakeBackend
from session_skills.brief import BEGIN, END, SELF_CHECK, brief
from session_skills.calibrate import CalibrationError, calibrate, score_question, stratified_sample
from session_skills.collect import build_list
from session_skills.questions import QUESTION_NAMES
from session_skills.repeated import normalise, repeated
from session_skills.state import State, StateError
from session_skills.synthetic import Conversation, write_claude_code, write_codex, write_opencode_db
from session_skills.tagging import AnswerError, build_prompt, parse_answer, tag


def answer_json(**overrides: float) -> str:
    data: dict[str, object] = {name: 0.0 for name in QUESTION_NAMES}
    data.update(task="coding", outcome="resolved", frustration=2)
    data.update(overrides)
    return json.dumps(data)


def seed_corpus(roots: dict[str, Path], secrets, days_ago: int = 3) -> None:
    day = TODAY - timedelta(days=days_ago)
    write_claude_code(roots["claude_code"], Conversation("a", day, [
        ("user", f"deploy with key {secrets.anthropic_key} from {secrets.home_mac}"),
        ("assistant", f"Done. Mail {secrets.email} if it fails."),
        ("user", "too long, keep it shorter"),
    ]))
    write_codex(roots["codex"], Conversation("b", day, [("user", "run the tests"), ("assistant", "ok"), ("user", "too long, shorter please")]))
    write_opencode_db(roots["opencode"] / "opencode.db", [
        Conversation("c", day, [("user", "run the tests please"), ("assistant", "ok"), ("user", "way too long")]),
    ])


# -- state ----------------------------------------------------------------------------------

def test_state_refuses_a_repo(tmp_path: Path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    with pytest.raises(StateError):
        State(tmp_path / "repo" / "state")


def test_build_list_redacts_and_is_private(cfg, state, roots, secrets):
    seed_corpus(roots, secrets)
    summary = build_list(cfg, state)
    assert summary.found == {"claude_code": 1, "codex": 1, "opencode": 1}
    assert summary.new == 3
    stored = "".join(p.read_text() for p in state.root.iterdir() if p.is_file())
    for value in (secrets.anthropic_key, secrets.email, secrets.home_user):
        assert value not in stored
    assert stat.S_IMODE(state.root.stat().st_mode) == 0o700
    for path in state.root.iterdir():
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    rows = state.tracker()
    assert {r["status"] for r in rows.values()} == {"new"}
    assert all(r["date"] == (TODAY - timedelta(days=3)).isoformat() for r in rows.values())

    again = build_list(cfg, state)
    assert again.unchanged == 3 and again.new == 0


def test_changed_session_is_retagged(cfg, state, roots, secrets):
    seed_corpus(roots, secrets)
    build_list(cfg, state)
    tag(cfg, state, FakeBackend(responder=lambda p: answer_json()))
    assert {r["status"] for r in state.tracker().values()} == {"tagged"}
    write_codex(roots["codex"], Conversation("b", TODAY - timedelta(days=3), [("user", "run the tests"), ("user", "and lint")]))
    summary = build_list(cfg, state)
    assert summary.changed == 1
    statuses = sorted(r["status"] for r in state.tracker().values())
    assert statuses == ["new", "tagged", "tagged"]
    assert len(state.tags()) == 2


# -- tagging ----------------------------------------------------------------------------------

def test_prompt_is_redacted_marked_and_json_only(cfg, state, roots, secrets):
    seed_corpus(roots, secrets)
    build_list(cfg, state)
    fake = FakeBackend(responder=lambda p: answer_json(too_long=0.9))
    summary = tag(cfg, state, fake)
    assert summary.tagged == 3 and summary.requests == 3
    prompts = "\n".join(fake.calls)
    assert secrets.anthropic_key not in prompts and secrets.email not in prompts
    assert all(p.startswith("[session-skills:worker-request]") for p in fake.calls)
    assert "exactly one JSON object" in fake.calls[0]
    assert all(r["flagged"] == "1" for r in state.tracker().values())


def test_request_budget_and_resume(cfg, state, roots, secrets):
    seed_corpus(roots, secrets)
    build_list(cfg, state)
    cfg.max_requests = 2
    first = tag(cfg, state, FakeBackend(responder=lambda p: answer_json()))
    assert first.tagged == 2 and first.remaining == 1
    second = tag(cfg, state, FakeBackend(responder=lambda p: answer_json()))
    assert second.tagged == 1 and second.remaining == 0


def test_failures_stay_blank_and_retry(cfg, state, roots, secrets):
    seed_corpus(roots, secrets)
    build_list(cfg, state)
    broken = FakeBackend(variant="broken")
    summary = tag(cfg, state, broken)
    assert summary.failed == 3 and summary.tagged == 0
    assert len(broken.calls) == 3 * cfg.retries
    assert state.tags() == {}  # blank, never zero
    assert {r["flagged"] for r in state.tracker().values()} == {""}

    attempts = {"n": 0}

    def flaky(prompt: str) -> str:
        attempts["n"] += 1
        return "not json" if attempts["n"] % 2 else answer_json(status_request=0.8)

    summary = tag(cfg, state, FakeBackend(responder=flaky))
    assert summary.tagged == 3 and summary.failed == 0
    assert all(t["scores"]["status_request"] == 0.8 for t in state.tags().values())


def test_long_chats_are_split_and_merged(cfg, state, roots):
    turns = [("user", f"message number {i} " + "word " * 40) for i in range(30)]
    write_claude_code(roots["claude_code"], Conversation("long", TODAY, turns))
    cfg.max_chars = 1000
    build_list(cfg, state)
    parts = int(next(iter(state.tracker().values()))["parts"])
    assert parts > 1
    replies = iter([answer_json(too_long=0.2, frustration=1)] + [answer_json(too_long=0.9, frustration=4)] * (parts - 1))
    summary = tag(cfg, state, FakeBackend(responder=lambda p: next(replies)))
    assert summary.requests == parts
    tagged = next(iter(state.tags().values()))
    assert tagged["scores"]["too_long"] == 0.9 and tagged["frustration"] == 4


def test_dry_run_sends_nothing(cfg, state, roots, secrets):
    seed_corpus(roots, secrets)
    build_list(cfg, state)
    fake = FakeBackend(responder=lambda p: answer_json())
    summary = tag(cfg, state, fake, dry_run=True)
    assert fake.calls == [] and summary.planned_chats == 3 and summary.requests == 3
    assert {r["status"] for r in state.tracker().values()} == {"new"}


def test_parse_answer_validation():
    parsed = parse_answer("Sure!\n```json\n" + answer_json(tone="40%", praise=True, task="weird") + "\n```")
    assert parsed["scores"]["tone"] == 0.4 and parsed["scores"]["praise"] == 1.0
    assert parsed["task"] == "other"
    with pytest.raises(AnswerError):
        parse_answer(json.dumps({"too_long": 0.5}))
    with pytest.raises(AnswerError):
        parse_answer(answer_json(too_long=1.7))
    assert "part 2 of 3" in build_prompt("USER: hi", 2, 3)


# -- calibration ------------------------------------------------------------------------------

def test_score_question():
    agree = [(0.9, 0.9)] * 5 + [(0.1, 0.0)] * 25
    result = score_question(agree, 0.7, 0.8)
    assert result.status == "reliable" and result.agreement == 1.0 and result.threshold == 0.7
    noisy = [(0.9, 0.0)] * 10 + [(0.1, 1.0)] * 10 + [(0.1, 0.0)] * 10
    assert score_question(noisy, 0.7, 0.8).status == "unreliable"
    assert score_question([(0.1, 0.0)] * 30, 0.7, 0.8).status == "untested"
    assert score_question([], 0.7, 0.8).status == "untested"
    shifted = [(0.5, 1.0)] * 10 + [(0.1, 0.0)] * 20
    assert score_question(shifted, 0.7, 0.8).threshold == 0.5


def test_stratified_sample_is_deterministic_and_spread():
    tags = {f"c{i:03d}": {"scores": {"x": i / 90}} for i in range(90)}
    first = stratified_sample(tags, 30, seed=1)
    assert first == stratified_sample(tags, 30, seed=1) and len(first) == 30
    low = sum(1 for cid in first if int(cid[1:]) < 30)
    high = sum(1 for cid in first if int(cid[1:]) >= 60)
    assert low == 10 and high == 10


def test_calibrate_requires_different_family(cfg, state, roots, secrets):
    seed_corpus(roots, secrets)
    build_list(cfg, state)
    with pytest.raises(CalibrationError):
        calibrate(cfg, state, FakeBackend())  # nothing tagged yet
    tag(cfg, state, FakeBackend())
    with pytest.raises(CalibrationError):
        calibrate(cfg, state, FakeBackend(), tagger_family="fake-agree")
    report = calibrate(cfg, state, FakeBackend(variant="noisy"), tagger_family="fake-agree")
    assert report["labelled"] == 3
    assert "tone" in report["unreliable"]
    assert state.read_json("calibration.json")["labeller"] == "fake"


# -- repeated asks ----------------------------------------------------------------------------

def test_normalise():
    assert normalise("Please run the tests!") == ("run", "tests")
    assert normalise("check src/app/main.py and https://example.org/x #123") == ("check", "path", "url", "n")


def test_repeated_clusters(cfg, state, roots):
    topics = ["parser", "cache", "exporter", "scheduler"]
    for i, text in enumerate(["run the tests", "Run the tests please", "run tests", "can you run the tests now?"]):
        write_claude_code(roots["claude_code"], Conversation(f"r{i}", TODAY - timedelta(days=i), [("user", text), ("user", f"rewrite the {topics[i]}")]))
    write_codex(roots["codex"], Conversation("s", TODAY, [("user", "status?")]))
    build_list(cfg, state)
    report = repeated(cfg, state)
    assert len(report["candidates"]) == 1
    top = report["candidates"][0]
    assert top["ask"] == "run tests" and top["sessions"] == 4 and top["occurrences"] == 4
    assert top["first"] == (TODAY - timedelta(days=3)).isoformat() and top["last"] == TODAY.isoformat()
    assert "hook" in top["suggestion"]


# -- brief ------------------------------------------------------------------------------------

def _corpus_for_brief(roots, secrets):
    recent = TODAY - timedelta(days=5)
    old = TODAY - timedelta(days=80)
    for i in range(3):
        write_claude_code(roots["claude_code"], Conversation(f"long{i}", recent - timedelta(days=i), [
            ("user", "explain the change"), ("assistant", "..."), ("user", f"too long, shorter please, send to {secrets.email}")]))
        write_codex(roots["codex"], Conversation(f"jar{i}", old, [("user", "too much jargon, plain english please")]))
    write_claude_code(roots["claude_code"], Conversation("tone1", recent, [("user", "your tone is preachy")]))


def test_brief_outputs(cfg, state, roots, secrets):
    _corpus_for_brief(roots, secrets)
    build_list(cfg, state)
    tag(cfg, state, FakeBackend())
    result = brief(cfg, state, TODAY)
    assert result["patterns"] == ["too_long"]
    assert result["stale"] == ["jargon"]
    assert result["skills"] == ["keep-replies-short"]
    out = state.root / "out"
    skill = (out / "skills" / "keep-replies-short" / "SKILL.md").read_text()
    assert skill.startswith("---\nname: keep-replies-short\ndescription: ")
    for heading in ("## When to use", "## What to do instead", "## Why", "## Examples"):
        assert heading in skill
    assert "[REDACTED:email]" in skill
    agents = (out / "AGENTS.md").read_text()
    assert agents.startswith(BEGIN) and agents.rstrip().endswith(END) and SELF_CHECK in agents
    assert "Lead with the answer" in (out / "CLAUDE.md.snippet").read_text()
    report = (out / "report.md").read_text()
    assert "| too_long | 3 |" in report and "jargon" in report
    everything = "".join(p.read_text() for p in out.rglob("*") if p.is_file())
    assert secrets.email not in everything
    assert stat.S_IMODE((out / "skills").stat().st_mode) == 0o700


def test_brief_excludes_unreliable_questions(cfg, state, roots, secrets):
    _corpus_for_brief(roots, secrets)
    build_list(cfg, state)
    tag(cfg, state, FakeBackend())
    state.write_json("calibration.json", {"questions": {"too_long": {"status": "unreliable", "threshold": 0.7}}, "unreliable": ["too_long"]})
    result = brief(cfg, state, TODAY)
    assert "too_long" not in result["patterns"] and result["excluded"] == ["too_long"]
    assert result["skills"] == []
