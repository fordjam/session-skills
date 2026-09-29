"""Command-line interface: ``session-skills <command>``.

Exit codes: 0 success; 1 finished but something failed (chats that could not be tagged, sources
that could not be read, labels that failed); 2 usage, configuration or state error; 3 the backend
could not be used or the run was stopped to protect the budget; 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from . import __version__
from .backends import Backend, FatalBackendError, make_backend
from .brief import brief
from .calibrate import CalibrationError, calibrate
from .collect import build_list, refresh_redaction
from .config import Config, SourceConfig, load_config
from .install import AGENTS, InstallError, InstallPlan, backups, install, plan_install, rollback
from .redact import redact
from .repeated import repeated
from .sources import ADAPTERS
from .state import State, StateError
from .synthetic import write_demo_sources
from .tagging import preview, tag


def _privacy_flags(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--local-only", action="store_true",
                     help="refuse any backend that is not provably on this machine (only local ollama passes)")
    sub.add_argument("--allow-remote", action="store_true",
                     help="let ollama use a non-loopback OLLAMA_HOST or a cloud model")


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
    t.add_argument("--max-requests", type=int,
                   help="hard cap on backend calls this run, retries and parts included (default 200; 0 = none)")
    t.add_argument("--dry-run", action="store_true", help="count requests and tokens; send nothing")
    t.add_argument("--preview", action="store_true",
                   help="print the exact command and payloads the next chat(s) would send; send nothing")
    t.add_argument("--limit", type=int, default=1, help="how many chats --preview shows (default 1)")
    _privacy_flags(t)

    c = sub.add_parser("calibrate", help="re-label a stratified sample with a second model family")
    c.add_argument("--backend", help="second backend, from a different model family than the tagger")
    c.add_argument("--sample-size", type=int)
    c.add_argument("--max-requests", type=int, help="hard cap on backend calls this run (default 100)")
    c.add_argument("--allow-same-family", action="store_true")
    _privacy_flags(c)

    sub.add_parser("repeated", help="cluster short repeated asks (no model)")

    r = sub.add_parser("brief", help="write report.md, skills, AGENTS.md block and CLAUDE.md snippet")
    r.add_argument("--today", help="YYYY-MM-DD (default: today)")

    i = sub.add_parser("install", help="install chosen skills (and optionally rules) for an agent; shows a diff first")
    i.add_argument("skills", nargs="*")
    i.add_argument("--all", action="store_true", help="every generated skill")
    i.add_argument("--agent", choices=AGENTS, default="claude", help="where to install (default claude)")
    i.add_argument("--target", help="override the skills directory")
    i.add_argument("--rules", action="store_true",
                   help="also write the rules block into the agent's CLAUDE.md / AGENTS.md (between markers)")
    i.add_argument("--dry-run", action="store_true", help="show the diff and stop")
    i.add_argument("--yes", action="store_true", help="do not prompt")
    i.add_argument("--force", action="store_true", help="overwrite skills that already exist with other content")

    rb = sub.add_parser("rollback", help="undo the latest install (or the one named)")
    rb.add_argument("backup", nargs="?")
    rb.add_argument("--list", action="store_true", help="list installs that can be rolled back")

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
    if getattr(args, "local_only", False):
        cfg.local_only = True
    if getattr(args, "allow_remote", False):
        cfg.allow_remote = True
    if cfg.local_only and cfg.allow_remote:
        raise ValueError("--local-only and --allow-remote cannot be combined")
    return cfg


def _backend(cfg: Config, spec: str) -> Backend:
    return make_backend(
        spec, timeout=cfg.timeout, allow_remote=cfg.allow_remote, local_only=cfg.local_only, pass_env=cfg.pass_env
    )


def _confirm(plan: InstallPlan) -> bool:
    _show_plan(plan)
    if not sys.stdin.isatty():
        print("stdin is not a terminal; re-run with --yes to confirm.", file=sys.stderr)
        return False
    return input("Apply these changes? [y/N] ").strip().lower() in {"y", "yes"}


def _show_plan(plan: InstallPlan) -> None:
    print(f"Agent: {plan.target.agent}. {len(plan.changes)} file(s) will change:")
    for change in plan.changes:
        print(f"\n# {change.label}")
        print(change.diff(), end="")
    print()


def _progress(message: str) -> None:
    print(f"  {message}", file=sys.stderr, flush=True)


def _tag(args: argparse.Namespace, cfg: Config, state: State) -> int:
    spec = args.backend or cfg.tag_backend
    if args.max_requests is not None:
        cfg.max_requests = args.max_requests
    backend = _backend(cfg, spec)
    if args.preview:
        items = preview(cfg, state, args.limit)
        info = backend.describe()
        if args.json:
            print(json.dumps({"backend": info, "payloads": items}, indent=2, sort_keys=True))
            return 0
        print(f"Backend: {spec} -> destination {info['destination']}; tools: {info['tools']}; "
              f"persistence: {info['persistence']}")
        if info["argv"]:
            print("Command: " + " ".join(json.dumps(a) if (not a or " " in a) else a for a in info["argv"]))
            print(f"Prompt via: {info['prompt_via']}; working directory: {info['cwd']}")
            print("Environment passed (names only): " + ", ".join(info["env"]))
        for note in info["notes"]:
            print(f"Note: {note}")
        for name, content in info["config_files"].items():
            print(f"Generated {name}: {content}")
        if not items:
            print("Nothing to send: every chat is tagged.")
        for item in items:
            print(f"\n===== chat {item['chat_id']} part {item['part']}/{item['parts']}: exact payload "
                  f"({len(item['payload'])} chars) =====")
            print(item["payload"], end="")
            print("===== end of payload =====")
        return 0
    ts = tag(cfg, state, backend, dry_run=args.dry_run, progress=None if args.json else _progress)
    data = {"backend": spec, **ts.as_dict()}
    if args.dry_run:
        _emit(args, data, [
            f"Would send {data['requests_this_run']} request(s) to {spec} this run "
            f"({data['planned_requests']} needed for {ts.planned_chats} chat(s); budget {ts.budget}).",
            f"Worst case with {cfg.retries} tries each: {data['max_attempts_this_run']} calls. "
            f"Estimated input tokens: ~{data['estimated_input_tokens']:,}.",
            "See exactly what would be sent with `session-skills tag --preview`.",
        ])
        return 0
    lines = [
        f"Sent {ts.requests} request(s) to {spec} (budget {ts.budget}; retries and parts count).",
        f"Tagged {ts.tagged}, failed {ts.failed}. Failed chats stay blank (never 0) and are retried next run.",
    ]
    for reason, count in sorted(ts.errors.items(), key=lambda kv: -kv[1]):
        lines.append(f"  failure x{count}: {reason}")
    if ts.budget_exhausted:
        lines.append("Stopped: request budget reached. Re-run to continue; finished parts are kept.")
    if ts.aborted:
        lines.append(f"Stopped early: {ts.aborted}")
    lines.append(f"Still untagged: {ts.remaining}")
    _emit(args, data, lines)
    if ts.aborted:
        if args.json:
            print(f"error: {ts.aborted}", file=sys.stderr)
        return 3
    return 1 if ts.failed else 0


def _calibrate(args: argparse.Namespace, cfg: Config, state: State) -> int:
    spec = args.backend or cfg.calibrate_backend
    if args.sample_size:
        cfg.sample_size = args.sample_size
    if args.max_requests is not None:
        cfg.calibrate_max_requests = args.max_requests
    try:
        report = calibrate(cfg, state, _backend(cfg, spec), allow_same_family=args.allow_same_family,
                           progress=None if args.json else _progress)
    except CalibrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    lines = [
        f"Labelled {report['labelled']}/{report['sample']} chats with {spec} ({report['requests']} requests).",
        "question                   n   agreement  kappa  threshold  status",
    ]
    for name, q in report["questions"].items():
        kappa = q["kappa"] if q["kappa"] is not None else "-"
        lines.append(f"{name:<26} {q['n']:<3} {q['agreement']:<10} {kappa!s:<6} {q['threshold']:<10} {q['status']}")
    lines.append(f"Excluded from the brief: {', '.join(report['unreliable']) or 'none'}")
    if report["label_failures"]:
        lines.append(f"Label failures: {report['label_failures']}")
    if report["stopped"]:
        lines.append(f"Stopped early: {report['stopped']}")
    _emit(args, report, lines)
    return 1 if report["label_failures"] or report["stopped"] else 0


def _install(args: argparse.Namespace, state: State) -> int:
    try:
        if args.dry_run:
            plan = plan_install(state, args.skills, args.all, agent=args.agent, target=args.target,
                                force=args.force, rules=args.rules)
            if not args.json:
                _show_plan(plan)
            backup_id = ""
        else:
            plan, backup_id = install(state, args.skills, all_skills=args.all, agent=args.agent, target=args.target,
                                      force=args.force, rules=args.rules,
                                      confirm=(lambda _plan: True) if args.yes else _confirm)
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    verb = "Would write" if args.dry_run else "Wrote"
    _emit(args, {
        "agent": plan.target.agent,
        "skills_dir": str(plan.target.skills_dir),
        "files": [str(c.path) for c in plan.changes],
        "copied": plan.copy,
        "skipped_existing": plan.skip_existing,
        "unchanged": plan.unchanged,
        "backup": backup_id,
        "dry_run": bool(args.dry_run),
    }, [
        f"{verb}: {', '.join(str(c.path) for c in plan.changes) or 'nothing'}",
        *([f"Skipped (already exist with other content; use --force): {', '.join(plan.skip_existing)}"]
          if plan.skip_existing else []),
        *([f"Undo with: session-skills rollback {backup_id}"] if backup_id else []),
    ])
    return 0


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
    state.lock()
    try:
        # every command, including install and rollback, first brings stored text up to the
        # current redaction policy (and runs the one-time cleanup for state from older versions)
        refresh_redaction(cfg, state)
        return _dispatch(args, cfg, state)
    finally:
        state.close()


def _dispatch(args: argparse.Namespace, cfg: Config, state: State) -> int:
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
        problems = [w for w in s.warnings if ": skipped " not in w]
        _emit(args, s.as_dict(), [
            "Sessions with human-typed text: " + ", ".join(f"{k} {v}" for k, v in s.found.items()),
            f"Tracker: {s.total} chats ({s.new} new, {s.changed} changed, {s.unchanged} unchanged"
            + (f", {s.duplicates} duplicate copies ignored" if s.duplicates else "") + ")",
            *([f"Re-redacted {s.re_redacted} stored chat(s) with the current policy."] if s.re_redacted else []),
            f"Untagged text: ~{s.chars // 4:,} input tokens. State: {state.root}",
            *[f"warning: {w}" for w in s.warnings],
        ])
        return 1 if problems else 0
    if args.command == "tag":
        return _tag(args, cfg, state)
    if args.command == "calibrate":
        return _calibrate(args, cfg, state)
    if args.command == "repeated":
        report = repeated(cfg, state)
        lines = [
            f"{c['sessions']:>3} sessions  {c['occurrences']:>3}x  {c['ask']}  -> {c['suggestion']}"
            for c in report["candidates"]
        ]
        _emit(args, report, lines or ["No repeated short asks yet."])
        return 0
    if args.command == "brief":
        today = date.fromisoformat(args.today) if args.today else None
        result = brief(cfg, state, today)
        _emit(args, result, [
            f"Report: {result['report']}",
            f"Patterns: {', '.join(result['patterns']) or 'none'}",
            f"Skills proposed: {', '.join(result['skills']) or 'none'} (nothing installed)",
            f"Automation candidates: {result['automation_candidates']}",
        ])
        return 0
    if args.command == "install":
        return _install(args, state)
    if args.command == "rollback":
        if args.list:
            names = backups(state)
            _emit(args, {"backups": names}, names or ["No installs to roll back."])
            return 0
        try:
            touched = rollback(state, args.backup)
        except InstallError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        _emit(args, {"restored": touched}, [f"Restored: {', '.join(touched)}"])
        return 0
    raise ValueError(f"unknown command {args.command}")


def _demo(args: argparse.Namespace) -> int:
    today = date.fromisoformat(args.today) if args.today else date.today()  # noqa: DTZ011
    root = Path(args.dir).expanduser() if args.dir else Path(tempfile.mkdtemp(prefix="session-skills-demo-"))
    cfg = Config(state_dir=root / "state")
    cfg.retry_delay = 0
    for name, paths in write_demo_sources(root / "sources", today).items():
        cfg.sources[name] = SourceConfig(True, paths)
    state = State(cfg.state_dir)
    try:
        build_list(cfg, state)
        tag(cfg, state, make_backend("fake:agree"))
        calibrate(cfg, state, make_backend("fake:noisy"))
        result = brief(cfg, state, today)
    finally:
        state.close()
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
    except FatalBackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 3
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        print("interrupted; finished work is saved", file=sys.stderr)
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()
