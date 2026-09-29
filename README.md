# session-skills

Mine your own AI coding-agent session history for:

- **repeated corrections** ("too long", "that flag doesn't exist", "did you run the tests?")
  that should become standing instructions and reusable **skills**, and
- **repeated asks** ("run the tests", "status?", "commit and push") that are **automation
  candidates**: a hook, a slash command, a script or a permission rule.

It reads Claude Code, OpenCode and Codex CLI sessions from your own disk, redacts them, has a
cheap model answer a fixed set of questions about each chat, checks those answers against a
second model from a different family, and writes a report plus proposed `SKILL.md` files, an
`AGENTS.md` block and a `CLAUDE.md` snippet. Nothing is installed unless you ask.

The method is borrowed from a widely shared idea: *your custom instructions are already in your
chat history*. Tag every chat with one cheap request, trust only patterns that show up in several
chats, and back every pattern with counts and a short quote.

## Install

Requires Python 3.11+. There are no runtime dependencies.

```sh
pipx install git+https://github.com/<you>/session-skills   # or: pip install .
session-skills --help
```

For tagging you need at least one worker CLI already logged in: `claude`, `codex`, `opencode` or
`ollama`. No API keys are read or stored by this tool.

## Quickstart

Try it first on a fictional corpus with offline fake models:

```sh
session-skills demo
```

Then on your own history:

```sh
session-skills build-list                 # find + redact sessions, write the tracker
session-skills tag --dry-run              # how many requests / tokens would be sent
session-skills tag                        # up to 200 requests per run; re-run to continue
session-skills calibrate                  # 30-chat sample re-labelled by a different model family
session-skills repeated                   # repeated short asks (no model)
session-skills brief                      # report.md, skills/, AGENTS.md, CLAUDE.md snippet
session-skills install keep-replies-short # copy chosen skills to ~/.claude/skills (asks first)
```

Every step is resumable. `build-list` can be re-run at any time: unchanged chats keep their tags,
changed chats are re-tagged, and chats whose files were later deleted by the agent's own cleanup
stay in the (redacted) state, so history accumulates across runs.

## Pipeline

| step | what it does | model? |
|---|---|---|
| `build-list` | Discovers sessions, keeps only human-typed messages (plus the first 600 chars of each assistant reply), redacts them, and writes `tracker.csv` + `chats.jsonl`. | no |
| `tag` | One request per chat (chats over 90k chars are split, taking the max per question). JSON-only answers, 3 retries, at most 200 requests per run. A failed chat stays blank, never 0, and is retried next run. | cheap model |
| `calibrate` | Stratified 30-chat sample (low / middle / high scores) labelled by a second backend from a **different model family**. Per question: agreement, Cohen's kappa, and the threshold that best matches. Unreliable questions are excluded from the brief. | second model |
| `repeated` | Deterministic clustering of short human messages across sessions: counts, sessions, first and last date, and an automation idea. | no |
| `brief` | Writes the report and proposed files. Patterns need at least 3 chats; patterns not seen in the last 60 days are dropped as stale. | no |
| `install` | Copies chosen skills after confirmation. Never edits `AGENTS.md` or `CLAUDE.md`. | no |

### The questions

Each chat gets a probability (0 to 1) for each of these, plus a task type, an outcome and a peak
frustration score (1 to 5):

`too_long`, `too_short`, `wrong_answer`, `ignored_instruction`, `format`, `tone`, `hedging`,
`made_up`, `asked_instead`, `repeated_myself`, `praise`, `claimed_without_checking`,
`acted_without_approval`, `status_request`, `approval_friction`, `jargon`.

## Sources

| source | default location | notes |
|---|---|---|
| Claude Code | `~/.claude/projects/**/*.jsonl` (honours `CLAUDE_CONFIG_DIR`) | Drops tool calls and results, `isMeta` turns (slash-command expansions), `isSidechain` turns and `subagents/` files (sub-agents), compaction summaries, and injected blocks such as `<system-reminder>`. For a slash command, only the typed arguments are kept. |
| Codex CLI | `~/.codex/sessions/**/*.jsonl` and `archived_sessions` (honours `CODEX_HOME`) | Uses `event_msg` / `user_message` records (exactly what was typed) when present, otherwise cleaned `response_item` user messages. Drops function calls, reasoning, AGENTS.md and environment context, and sub-agent rollouts. Older rollouts without the envelope are supported. |
| OpenCode | `~/.local/share/opencode/opencode*.db` (honours `XDG_DATA_HOME`, `OPENCODE_DB`) and legacy `storage/` | See assumptions below. |

Override any path with `--path SOURCE=GLOB` or in the config file. One session is one chat.
Sessions with no human-typed text are skipped, and so are sessions created by this tool's own
worker requests.

**OpenCode assumptions.** OpenCode's storage layout was read from its source, not from real
sessions. Current releases keep sessions in SQLite, in `session`, `message` and `part` tables:
`message.data` is JSON with `role`, `part.data` is JSON with `type` / `text` / `synthetic`, and
`session.parent_id` marks sub-agent sessions, which are skipped. Only non-synthetic `text` parts
are kept. Older releases used JSON files under `storage/session|message|part/`, and those are
read too. OpenCode does not mark slash-command expansions, so a custom command's template can
show up as typed text. If your layout differs, point `sources.opencode.paths` at the right
database or directory, or open an issue.

## Privacy model

- **Local first.** Session files are opened read-only. Nothing is uploaded except the redacted
  text of each chat, sent to the worker CLI *you* chose. Use `ollama:MODEL` to keep everything on
  your machine.
- **Deterministic redaction before storage and before any model call.** The redactor
  (`session_skills/redact.py`) replaces API keys and tokens (known prefixes, JWTs, bearer values,
  and long high-entropy strings), email addresses, runs of 12 to 19 digits (cards, accounts),
  money amounts (`$N`), home-directory user names (`/Users/<name>` becomes `~`), `.env` contents,
  URL credentials, private keys, and pasted blocks (`<pasted_content>`, long code fences, long
  runs of log lines). It is covered by tests that use synthetic secrets. Check it on your own
  text with `echo "..." | session-skills redact`.
- **Names are not detected automatically.** Add people, customers and project code names to
  `redact.deny_words`, or add regexes to `redact.extra_patterns`.
- **Private state outside any repository.** State lives in `~/.local/state/session-skills`
  (override with `--state-dir` or `SESSION_SKILLS_STATE`). It is created `0700`, its files are
  written `0600`, and the tool refuses a state directory inside a git, hg or svn work tree. Only
  redacted text is stored. Session paths are stored only as hashes.
- **Workers are sandboxed from your project.** Each worker runs in a fresh empty temp directory.
  `claude` runs with no tools and without saving a session. `codex` runs read-only and ephemeral.
  Every prompt starts with a marker, so any worker session that does get saved is skipped by the
  next `build-list`.
- **Nothing is applied automatically.** `brief` only writes into the state directory. `install`
  lists what it will copy and asks first. It refuses to run without a terminal unless you pass
  `--yes`.

Redaction is best-effort pattern matching. Read `out/report.md` and the generated skills before
you share them.

## Backends and costs

| spec | runs | family |
|---|---|---|
| `claude:haiku` (default tagger) | `claude -p --model haiku` | anthropic |
| `codex[:MODEL]` (default calibrator) | `codex exec [-m MODEL]` | openai |
| `opencode:PROVIDER/MODEL` | `opencode run -m PROVIDER/MODEL` | inferred from the model name |
| `ollama:MODEL` | `ollama run MODEL` | inferred from the model name |
| `exec:COMMAND` | any command that reads the prompt on stdin and prints JSON | custom |
| `fake[:agree\|noisy\|broken]` | offline heuristic for tests and the demo | fake |

Use a cheap, fast model for `tag`: it is one short classification per chat. `calibrate` only
labels 30 chats, so a stronger model is fine there. It has to be a different family from the
tagger (pass `--allow-same-family` to override).

Cost is roughly one request per chat. Each request carries about 700 tokens of instructions plus
the chat's redacted text, which is human messages plus 600 chars per assistant reply, so it is
usually a few thousand tokens. Run `session-skills tag --dry-run` to see the exact request count
and estimated input tokens before you spend anything. Subscription CLIs count this against your
plan's usage limits. `ollama` is free but slower and less accurate.

## Configuration

Optional, at `~/.config/session-skills/config.toml` (or `--config`). See
[`examples/config.toml`](examples/config.toml):

```toml
[tag]
backend = "claude:haiku"
max_requests = 200

[calibrate]
backend = "codex"

[redact]
deny_words = ["Initech", "Project Falcon"]
```

## Sample output

From `session-skills demo` (fictional data, fake models):

```markdown
## Repeated corrections

| pattern | chats | % | last seen | quote |
|---|---:|---:|---|---|
| claimed_without_checking | 3 | 17.6% | 2026-09-23 | "did you run the tests? it still fails on CI" |
| ignored_instruction | 3 | 17.6% | 2026-09-20 | "I said no new dependencies. remove it" |
| made_up | 3 | 17.6% | 2026-09-25 | "that flag doesn't exist in this version, check the help output" |
| too_long | 3 | 17.6% | 2026-09-27 | "way too long, just show me the diff" |

## Automation candidates (repeated asks)

| ask | example | sessions | times | first | last | idea |
|---|---|---:|---:|---|---|---|
| run tests | run the tests | 5 | 5 | 2026-08-24 | 2026-09-27 | hook that runs the check automatically after edits |
| status | status? | 4 | 4 | 2026-08-20 | 2026-09-25 | status digest: a hook or scheduled summary |
| commit push | commit and push | 3 | 3 | 2026-09-04 | 2026-09-20 | slash command or script for the git flow |
```

A generated skill, `out/skills/check-that-it-exists/SKILL.md`:

```markdown
---
name: check-that-it-exists
description: "Use before naming an API, flag, file, package or source. The user caught invented ones."
---

# Check that it exists

## When to use

Use before naming an API, flag, file, package or source. The user caught invented ones.

## What to do instead

- Look it up in the codebase or docs before naming it.
- If you cannot confirm it, say it is unconfirmed.

## Why

Invented details cost time to track down and eroded trust. This came up in 3 of 17 reviewed
chats (17.6%), most recently 2026-09-25.

## Examples

What the user said (redacted):

- "that flag doesn't exist in this version, check the help output"
- "that function doesn't exist. check the code before naming things"
```

`out/AGENTS.md` contains a block between `<!-- session-skills:begin -->` and
`<!-- session-skills:end -->` that you can paste into an `AGENTS.md` for OpenCode or Codex.
`out/CLAUDE.md.snippet` has the same rules for `CLAUDE.md`.

## Limitations

- The questions are fixed. Adding one means editing `questions.py` (see CONTRIBUTING).
- Repeated-ask clustering is lexical (token overlap), so paraphrases with no shared words are not
  merged.
- Quotes are picked by keyword cues. When no short message matches, the table shows `—`.

## License

MIT
