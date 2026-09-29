"""Command-line interface: ``session-skills <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .backends import make_backend
from .brief import brief
from .calibrate import CalibrationError, calibrate
from .collect import build_list
from .config import Config, SourceConfig, load_config
from .install import DEFAULT_TARGET, InstallError, InstallPlan, install
from .redact import redact
from .repeated import repeated
from .sources import ADAPTERS
from .state import State, StateError
from .synthetic import write_demo_sources
from .tagging import tag


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="session-skills",
        description="Mine your own agent session history for repeated corrections and repeated asks.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--config", help="TOML config (default ~/.config/session-skills/config.toml if present)")
    p.add_argument("--state-dir", help="private state directory (default ~/.local/state/session-skills)")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-list", help="discover and redact sessions; update the tracker")
    b.add_argument("--source", action="append", choices=sorted(ADAPTERS), help="only this source (repeatable)")
    b.add_argument("--path", action="append", default=[], metavar="SOURCE=GLOB",
                   help="override a source's paths, e.g. codex=~/backup/codex/**/*.jsonl (repeatable)")

    t = sub.add_parser("tag", help="tag untagged chats with the cheap backend (resumable)")
    t.add_argument("--backend", help="e.g. claude:haiku, codex:MODEL, opencode:PROVIDER/MODEL, ollama:MODEL, fake")
    t.add_argument("--max-requests", type=int, help="request budget for this run (default 200)")
    t.add_argument("--dry-run", action="store_true", help="show what would be sent and estimated tokens")

    c = sub.add_parser("calibrate", help="re-label a stratified sample with a second model family")
    c.add_argument("--backend", help="second backend, from a different model family than the tagger")
    c.add_argument("--sample-size", type=int)
    c.add_argument("--allow-same-family", action="store_true")

    sub.add_parser("repeated", help="cluster short repeated asks (no model)")

    r = sub.add_parser("brief", help="write report.md, skills, AGENTS.md block and CLAUDE.md snippet")
    r.add_argument("--today", help="YYYY-MM-DD (default: today)")

    i = sub.add_parser("install", help="copy chosen generated skills (asks first)")
    i.add_argument("skills", nargs="*")
    i.add_argument("--all", action="store_true")
    i.add_argument("--target", default=DEFAULT_TARGET, help=f"skills directory (default {DEFAULT_TARGET})")
    i.add_argument("--yes", action="store_true", help="do not prompt")
    i.add_argument("--force", action="store_true", help="overwrite skills that already exist")

    sub.add_parser("redact", help="redact stdin to stdout (to check the redactor)")

    d = sub.add_parser("demo", help="run the whole pipeline on a fictional corpus with fake backends")
    d.add_argument("--dir", help="where to write the demo corpus and state (default: a new temp dir)")
    d.add_argument("--today", help="YYYY-MM-DD (default: today)")
    return p


def _emit(args: argparse.Namespace, data: dict[str, Any], lines: Sequence[str]) -> None:
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        for line in lines:
            print(line)


def _config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    if args.state_dir:
        cfg.state_dir = Path(args.state_dir).expanduser()
    return cfg


def _confirm(plan: InstallPlan) -> bool:
    print(f"Copy {len(plan.copy)} skill(s) into {plan.target}:")
    for name in plan.copy:
        print(f"  - {name}")
    if not sys.stdin.isatty():
        print("stdin is not a terminal; re-run with --yes to confirm.", file=sys.stderr)
        return False
    return input("Proceed? [y/N] ").strip().lower() in {"y", "yes"}


def _progress(message: str) -> None:
    print(f"  {message}", file=sys.stderr, flush=True)


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "redact":
        opts = load_config(args.config).redact_options()
        sys.stdout.write(redact(sys.stdin.read(), opts))
        return 0
    if args.command == "demo":
        return _demo(args)
    cfg = _config(args)
    state = State(cfg.state_dir)

    if args.command == "build-list":
        overrides: dict[str, list[str]] = {}
        for spec in args.path:
            name, _, glob = spec.partition("=")
            if name not in cfg.sources or not glob:
                raise ValueError(f"--path expects SOURCE=GLOB with SOURCE in {sorted(ADAPTERS)}")
            overrides.setdefault(name, []).append(glob)
        for name, globs in overrides.items():
            cfg.sources[name] = SourceConfig(True, globs)
        s = build_list(cfg, state, args.source or ())
        _emit(args, s.as_dict(), [
            "Sessions with human-typed text: " + ", ".join(f"{k} {v}" for k, v in s.found.items()),
            f"Tracker: {s.total} chats ({s.new} new, {s.changed} changed, {s.unchanged} unchanged)",
            f"Untagged text: ~{s.chars // 4:,} input tokens. State: {state.root}",
        ])
    elif args.command == "tag":
        spec = args.backend or cfg.tag_backend
        if args.max_requests is not None:
            cfg.max_requests = args.max_requests
        ts = tag(cfg, state, make_backend(spec, timeout=cfg.timeout), dry_run=args.dry_run,
                 progress=None if args.json else _progress)
        verb = "Would send" if args.dry_run else "Sent"
        detail = (
            f"Planned chats: {ts.planned_chats}; estimated input tokens: ~{ts.as_dict()['estimated_input_tokens']:,}"
            if args.dry_run
            else f"Tagged {ts.tagged}, failed {ts.failed} (failed chats stay blank and are retried next run)."
        )
        _emit(args, {"backend": spec, **ts.as_dict()}, [
            f"{verb} {ts.requests} request(s) to {spec}.", detail, f"Still untagged: {ts.remaining}",
        ])
    elif args.command == "calibrate":
        spec = args.backend or cfg.calibrate_backend
        if args.sample_size:
            cfg.sample_size = args.sample_size
        taggers = {str(t.get("backend", "")) for t in state.tags().values()}
        tagger_family = make_backend(sorted(taggers)[0]).family() if len(taggers) == 1 and "" not in taggers else None
        try:
            report = calibrate(cfg, state, make_backend(spec, timeout=cfg.timeout), tagger_family=tagger_family,
                               allow_same_family=args.allow_same_family, progress=None if args.json else _progress)
        except CalibrationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        lines = [f"Labelled {report['labelled']}/{report['sample']} chats with {spec}.", "question  agreement  kappa  threshold  status"]
        for name, q in report["questions"].items():
            lines.append(f"{name:<26} {q['agreement']:<9} {q['kappa'] if q['kappa'] is not None else '-':<6} {q['threshold']:<9} {q['status']}")
        lines.append(f"Excluded from the brief: {', '.join(report['unreliable']) or 'none'}")
        _emit(args, report, lines)
    elif args.command == "repeated":
        report = repeated(cfg, state)
        lines = [f"{c['sessions']:>3} sessions  {c['occurrences']:>3}x  {c['ask']}  -> {c['suggestion']}" for c in report["candidates"]]
        _emit(args, report, lines or ["No repeated short asks yet."])
    elif args.command == "brief":
        today = date.fromisoformat(args.today) if args.today else None
        result = brief(cfg, state, today)
        _emit(args, result, [
            f"Report: {result['report']}",
            f"Patterns: {', '.join(result['patterns']) or 'none'}",
            f"Skills proposed: {', '.join(result['skills']) or 'none'} (nothing installed)",
            f"Automation candidates: {result['automation_candidates']}",
        ])
    elif args.command == "install":
        try:
            plan = install(state, args.skills, all_skills=args.all, target=args.target, force=args.force,
                           confirm=(lambda _plan: True) if args.yes else _confirm)
        except InstallError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        _emit(args, {"target": str(plan.target), "copied": plan.copy, "skipped_existing": plan.skip_existing}, [
            f"Copied: {', '.join(plan.copy) or 'nothing'} -> {plan.target}",
            *( [f"Skipped (already exist; use --force): {', '.join(plan.skip_existing)}"] if plan.skip_existing else [] ),
        ])
    return 0


def _demo(args: argparse.Namespace) -> int:
    today = date.fromisoformat(args.today) if args.today else date.today()
    root = Path(args.dir).expanduser() if args.dir else Path(tempfile.mkdtemp(prefix="session-skills-demo-"))
    cfg = Config(state_dir=root / "state")
    for name, paths in write_demo_sources(root / "sources", today).items():
        cfg.sources[name] = SourceConfig(True, paths)
    state = State(cfg.state_dir)
    build_list(cfg, state)
    tag(cfg, state, make_backend("fake:agree"))
    calibrate(cfg, state, make_backend("fake:noisy"), tagger_family="fake-agree")
    result = brief(cfg, state, today)
    report = Path(result["report"]).read_text(encoding="utf-8")
    if args.json:
        print(json.dumps({**result, "demo_dir": str(root)}, indent=2, sort_keys=True))
    else:
        print(report)
        print(f"(demo files in {root})")
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    try:
        code = run(argv)
    except StateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        print("interrupted; progress so far is saved", file=sys.stderr)
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()
