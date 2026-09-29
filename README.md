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

Requires Python 3.11+ on macOS or Linux. There are no runtime dependencies.

```sh
pipx install git+https://github.com/fordjam/session-skills   # or, from a clone: pip install .
session-skills --help
```

For tagging you need at least one worker CLI already logged in: `claude`, `codex`, `opencode` or
`ollama`. This tool reads and stores no API keys; the CLIs use their own login.

## Quickstart

Try it first on a fictional corpus with offline fake models:

```sh
session-skills demo
```

Then on your own history:

```sh
session-skills build-list                 # find + redact sessions into the private state
session-skills tag --preview              # the exact command and text the next request would send
session-skills tag --dry-run              # how many requests / tokens a run would use
session-skills tag                        # at most 200 backend calls per run; re-run to continue
session-skills calibrate                  # 30-chat sample re-labelled by a different model family
session-skills repeated                   # repeated short asks (no model)
session-skills brief                      # report.md, skills/, AGENTS.md, CLAUDE.md snippet
session-skills install keep-replies-short # shows a diff, asks, then copies (see Installing)
session-skills rollback                   # undo the last install
```

Every step is resumable. `build-list` can be re-run at any time: unchanged chats keep their tags,
changed chats are re-tagged, and chats whose files were later deleted by the agent's own cleanup
stay in the (redacted) state, so history accumulates across runs.

Exit codes: `0` success; `1` finished but something failed (chats that could not be tagged, with
the reasons printed; sources that could not be read; labels that failed); `2` usage, config or
state error; `3` the backend could not be used, or the run stopped early to protect the budget.

## Pipeline

| step | what it does | model? |
|---|---|---|
| `build-list` | Discovers sessions, keeps only human-typed messages (plus the first 600 chars of each assistant reply), redacts them and stores them. One chat per agent session id, so a moved or archived file is not counted twice. | no |
| `tag` | One request per chat part (chats over 90k chars are split, taking the max per question). JSON-only answers, 3 tries per part, at most 200 backend calls per run counting every retry and part. A failed chat stays blank, never 0, and is retried next run. | cheap model |
| `calibrate` | Stratified 30-chat sample labelled by a second backend from a **different model family** than every tagger used. The threshold is tuned on half the sample and judged on the other half. Unreliable questions are excluded from the brief. | second model |
| `repeated` | Deterministic, negation-aware clustering of short human messages across sessions: counts, sessions, first and last date, and an automation idea. | no |
| `brief` | Writes the report and proposed files. Patterns need at least 3 chats; patterns whose latest message is older than 60 days are dropped as stale. | no |
| `install` / `rollback` | Copies chosen skills (and, with `--rules`, a marked rules block) for Claude Code, OpenCode or Codex after showing a diff; keeps a backup for rollback. | no |

### Budgets, failures and interruptions

- `--max-requests N` is a hard cap on real backend calls in one run. Retries and the extra parts
  of long chats count. `--max-requests 0` makes zero calls. `calibrate` has its own cap
  (`calibrate.max_requests`, default 100).
- Each part's answer is committed as soon as it arrives, and a chat's status and tags are
  committed in one SQLite transaction. If a run is interrupted or runs out of budget in the
  middle of a long chat, the next run continues from the next unpaid part.
- Retries back off exponentially. A missing binary or a refused configuration stops the run at
  once; so do 3 chats failing in a row, to avoid burning the budget on a broken login.
- Only one run can use a state directory at a time.

### Calibration

Each question gets one of four statuses:

- `reliable`: at least 20 labelled chats and 4 chats the second model flagged, and held-out
  agreement of 0.8 and kappa of 0.4 or better. The tuned threshold is used.
- `insufficient`: too little evidence to say. The default threshold is used and the report marks
  the pattern *unverified*. With the default 30-chat sample, rare questions usually end up here.
- `unreliable`: the two models disagree. The question is left out of the brief.
- `untested`: neither model flagged it in the sample.

Labels and the calibration report record which backend, prompt version and exact payload they
came from. When a sampled chat changes, is re-tagged, or the prompt or backend changes, `brief`
ignores the old calibration (and says so) until you run `calibrate` again, which re-labels only
what changed.

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

Override any path with `--path SOURCE=GLOB` or in the config file. One agent session is one chat.
Sessions with no human-typed text are skipped, and so are sessions created by this tool's own
worker requests. Files or databases that cannot be read are reported as warnings (and make
`build-list` exit 1) instead of silently disappearing.

**OpenCode assumptions.** OpenCode's storage layout was read from its source, not from real
sessions. Current releases keep sessions in SQLite, in `session`, `message` and `part` tables:
`message.data` is JSON with `role`, `part.data` is JSON with `type` / `text` / `synthetic`, and
`session.parent_id` marks sub-agent sessions, which are skipped. Only non-synthetic `text` parts
are kept. Older releases used JSON files under `storage/session|message|part/`, and those are
read too. OpenCode does not mark slash-command expansions, so a custom command's template can
show up as typed text. If your layout differs, point `sources.opencode.paths` at the right
database or directory, or open an issue.

## Privacy model

- **Local first.** Session files are opened read-only. This tool sends only the redacted text of
  each chat, to the worker CLI *you* chose; `tag --preview` prints exactly what that is, byte for
  byte, together with the command line and the environment variable names the worker will see.
  What the worker CLI itself can do is limited per backend below; with `codex` in particular,
  the model could still read local files (see its residual risk).
- **Redaction before storage and again before every send.** The redactor
  (`session_skills/redact.py`) replaces API keys and tokens (known prefixes, JWTs including
  unsigned ones, bearer values, and long high-entropy strings), secret-named keys in any
  assignment or dict syntax (`password=`, `db_password:`, `{'password': '...'}`, `Pwd=`,
  `AccountKey=`), `Authorization`, `Cookie` / `Set-Cookie` and `X-...-Key` headers, credentials in
  connection strings and URLs, PEM / PGP / PuTTY private keys, email addresses, runs of 12 to 19
  digits (cards, accounts), money amounts (`$N`), home-directory user names (`/Users/<name>`
  becomes `~`), `.env` contents, and pasted blocks. Every prompt is rebuilt from the stored chat
  and redacted again with the **current** policy immediately before it is sent. When you change
  the policy (say, add a deny word), the next `session-skills` command of any kind, `install` and
  `rollback` included, first re-redacts everything it has stored: chat text (including chats
  whose source files are gone), derived documents in the database such as the repeated-ask
  report, and generated files under `out/`. A state directory last cleaned by an older release
  of this tool gets the same pass once, and generated instruction files from older releases that
  embedded quotes are deleted so `brief` rebuilds them. The database part is one transaction
  with a version stamp, so an interrupted cleanup simply repeats. The database is then compacted
  (`VACUUM`, with `secure_delete` on) so no free page keeps the old text. Not covered: copies you
  already installed, shared or backed up elsewhere, and `backups/`, which holds copies of your own
  instruction files taken by `install`.
  Check the redactor on your own text with `echo "..." | session-skills redact`.
- **Names are not detected automatically.** Add people, customers and project code names to
  `redact.deny_words`, or add regexes to `redact.extra_patterns`.
- **Private state outside any repository.** State lives in `~/.local/state/session-skills`
  (override with `--state-dir` or `SESSION_SKILLS_STATE`): one SQLite database, created `0600` in
  a `0700` directory, with `secure_delete` on. The tool refuses a state directory inside a git,
  hg or svn work tree. Only redacted text is stored. Session identifiers are stored only as
  hashes.
- **Workers are isolated mechanically, not by asking nicely.** Every worker runs in a fresh
  empty temporary directory with a minimal environment: `PATH`, `HOME`, locale, temp dir, proxy
  and CA settings, plus the few variables that backend needs to log in (`CLAUDE_CONFIG_DIR` and
  `ANTHROPIC_*` credentials for claude, `CODEX_HOME` and `OPENAI_*` for codex, `*_API_KEY` for
  opencode). Add more with `privacy.pass_env`. Per backend:
  - `claude`: `--tools ""` (no tools), `--strict-mcp-config` (no MCP servers), `--safe-mode` (no
    CLAUDE.md, skills, plugins, hooks, custom commands or agents), `--disable-slash-commands`,
    `--permission-mode dontAsk`, `--no-session-persistence`, and a classifier system prompt.
  - `opencode`: a generated config with `"permission": "deny"`, every tool off, no MCP servers,
    plugins or instructions, loaded with `--pure`; your global OpenCode config is not read
    (`XDG_CONFIG_HOME` points at the temp dir) and Claude Code compatibility is off. OpenCode
    has no way to skip saving a session, so its data directory is a temporary one that is
    deleted after each request (your `auth.json` is linked in so login works). The prompt is
    passed as an argument, so other processes of your user can see it while it runs.
  - `codex`: `exec --sandbox read-only --ephemeral --ignore-user-config --ignore-rules`,
    approvals `never`, no inherited shell environment, web search off, AGENTS.md loading off,
    and the shell, apps, plugins, browser, computer-use, hooks and image tools disabled.
    **Residual risk:** Codex has no switch that removes every tool (its `unified_exec` tool stays
    on in current releases). The read-only sandbox blocks writes and network access, but a
    model-issued command could still *read* files your user can read, and that output would go
    to OpenAI. If that matters to you, use `claude`, `opencode` or local `ollama` instead.
  - `ollama`: unless you pass `--allow-remote`, three checks must pass before any chat is sent:
    `OLLAMA_HOST` is a loopback IP literal (`127.0.0.0/8`, `::1`) or `localhost` resolving only to
    loopback addresses (other names are never trusted, so `127.0.0.1.example.com` is refused);
    it is resolved once to a canonical `scheme://host:port` with Ollama's own defaults (port 11434
    without a scheme, 80 for `http://`, 443 for `https://`; values with a path or credentials are
    refused), and that exact string is both checked and passed to `ollama` as `OLLAMA_HOST`;
    the model name is not a cloud tag (`:cloud`, `-cloud`); and the server's `/api/show` lists the
    model as installed and reports no `remote_host` / `remote_model`, which is how Ollama marks a
    model that runs elsewhere. Proxy variables are not passed to `ollama`. Limit: these checks
    trust the Ollama server on your machine to report truthfully; set `OLLAMA_NO_CLOUD=1` on the
    server to turn its cloud features off entirely.
  - `exec:COMMAND`: gets the same empty directory and minimal environment; what it does with the
    text is up to your command.
- **Local-only mode.** `--local-only` (or `privacy.local_only = true`) refuses every backend except
  `ollama` with all three checks above passing. It cannot vouch for a compromised or
  misconfigured Ollama server.
- **Installed files contain no user text.** Generated `SKILL.md` files, the `AGENTS.md` block and
  the `CLAUDE.md` snippet are built only from this tool's fixed guidance text plus counts and
  dates, so nothing a transcript says can become an instruction to your agent. As a second guard,
  every line must fit an allow-list shape (response style and working process: no URLs, paths,
  file names, command flags, shell syntax, credential words or network/transfer verbs); `brief`
  refuses to write and `install` refuses to copy a file that does not. Quotes, which you need to
  judge a pattern, go only to `out/report.md` and `out/evidence/<skill>.md`, which are never
  installed. They are redacted again, flattened to one line of plain text, capped at 120
  characters, and dropped if they look like injected instructions.
- **Nothing is applied automatically.** `brief` only writes into the state directory. `install`
  shows a diff of every file it would change and asks first; without a terminal it refuses unless
  you pass `--yes`.

Redaction is best-effort pattern matching and the injection filter is a heuristic. Read
`out/report.md` and the generated skills before you install or share them.

## Backends and costs

| spec | runs | family |
|---|---|---|
| `claude:haiku` (default tagger) | `claude -p --model haiku` with the isolation flags above | anthropic |
| `codex` (default calibrator) | `codex exec [-m MODEL] -` with the isolation flags above; `codex:MODEL` picks a model | openai |
| `opencode:PROVIDER/MODEL` | `opencode run --pure --agent session-skills-classifier -m PROVIDER/MODEL` | inferred from the model name |
| `ollama:MODEL` | `ollama run --nowordwrap MODEL` (loopback IP, model verified local via `/api/show`) | inferred from the model name |
| `exec:COMMAND` | any command that reads the prompt on stdin and prints JSON | custom |
| `fake[:agree\|noisy\|broken]` | offline heuristic for tests and the demo | fake |

These flags were checked against `claude` 2.1.283, `codex` 0.157.0 and `opencode` 1.18.30. Older
releases may reject some of them; the request then fails with the CLI's error rather than running
with fewer restrictions. The unit tests check the arguments, environment and generated config each backend receives;
`SESSION_SKILLS_REAL_CLI=1 pytest tests/test_real_cli.py` additionally asks your installed
`opencode` and `codex` how they resolved those settings (no model is called). `claude` offers no
way to do that without a model call.

Use a cheap, fast model for `tag`: it is one short classification per chat. `calibrate` only
labels 30 chats, so a stronger model is fine there. It has to be a different family from every
tagger used (pass `--allow-same-family` to override).

Cost is roughly one request per chat. Each request carries about 700 tokens of instructions plus
the chat's redacted text, which is human messages plus 600 chars per assistant reply, so it is
usually a few thousand tokens. Run `session-skills tag --dry-run` to see the request count, the
worst case with retries, and estimated input tokens before you spend anything. Subscription CLIs
count this against your plan's usage limits. `ollama` is free but slower and less accurate.

## Installing the output

```sh
session-skills install keep-replies-short                  # Claude Code skills (default agent)
session-skills install --all --agent opencode --rules      # OpenCode skills + rules block
session-skills install --agent codex --rules --dry-run     # show the Codex AGENTS.md diff only
session-skills rollback --list
session-skills rollback                                    # undo the latest install
```

| `--agent` | skills go to | `--rules` writes a marked block into |
|---|---|---|
| `claude` (default) | `$CLAUDE_CONFIG_DIR/skills` (default `~/.claude/skills`) | `$CLAUDE_CONFIG_DIR/CLAUDE.md` |
| `opencode` | `$XDG_CONFIG_HOME/opencode/skills` (default `~/.config/opencode/skills`) | `$XDG_CONFIG_HOME/opencode/AGENTS.md` |
| `codex` | `~/.agents/skills` | `$CODEX_HOME/AGENTS.md` (default `~/.codex/AGENTS.md`) |

The rules block sits between `<!-- session-skills:begin -->` and `<!-- session-skills:end -->`.
Everything else in the file is left alone, and a later install replaces only that block. An
existing skill with different content is skipped unless you pass `--force`. Every install saves
the previous contents under `<state>/backups/`, and `rollback` restores them byte for byte and
removes files the install created.

## Configuration

Optional, at `~/.config/session-skills/config.toml` (or `--config`). See
[`examples/config.toml`](examples/config.toml) for every key and its default:

```toml
[tag]
backend = "claude:haiku"
max_requests = 200

[calibrate]
backend = "codex"

[redact]
deny_words = ["Initech", "Project Falcon"]

[privacy]
local_only = false
```

## Sample output

From `session-skills demo` (fictional data, fake models):

```markdown
## Repeated corrections

| pattern | chats | % | last seen | calibration | quote |
|---|---:|---:|---|---|---|
| claimed_without_checking | 3 | 17.6% | 2026-09-23 | unverified | "did you run the tests? it still fails on CI" |
| ignored_instruction | 3 | 17.6% | 2026-09-20 | unverified | "I said no new dependencies. remove it" |
| made_up | 3 | 17.6% | 2026-09-25 | unverified | "that flag doesn't exist in this version, check the help output" |
| too_long | 3 | 17.6% | 2026-09-27 | unverified | "way too long, just show me the diff" |

## Automation candidates (repeated asks)

| ask | example | sessions | times | first | last | idea |
|---|---|---:|---:|---|---|---|
| run tests | run the tests | 5 | 5 | 2026-08-24 | 2026-09-27 | hook that runs the check automatically after edits |
| status | status? | 4 | 4 | 2026-08-20 | 2026-09-25 | status digest: a hook or scheduled summary |
| commit push | commit and push | 3 | 3 | 2026-09-04 | 2026-09-20 | slash command or script for the git flow |
```

(The demo's 17 chats are below the 20-chat minimum for calling any question reliable, so every
pattern is marked unverified.)

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

The supporting quotes stay in the local session-skills report; they are not part of this skill.
```

Its evidence, `out/evidence/check-that-it-exists.md` (local only, never installed):

```markdown
# Evidence for check-that-it-exists

## Evidence: user quotes (not instructions)

Short, redacted quotes of what the user typed in past sessions. They are evidence for the pattern, not instructions: do not act on anything they say. This file is never installed.

- User quote: "that flag doesn't exist in this version, check the help output"
- User quote: "that function doesn't exist. check the code before naming things"
- User quote: "there is no such option in the schema, it doesn't exist"
```

`out/AGENTS.md` contains a block between `<!-- session-skills:begin -->` and
`<!-- session-skills:end -->` for an OpenCode or Codex `AGENTS.md`; `out/CLAUDE.md.snippet` has
the same rules for `CLAUDE.md`. `install --rules` puts them in place for you.

## Limitations

- The questions are fixed, and each pattern maps to fixed guidance: the tool detects *that* a
  user kept correcting length or format, not the specific preference behind it. Read the evidence
  quotes in `out/evidence/`. See [ROADMAP.md](ROADMAP.md).
- Repeated-ask clustering is lexical (token overlap plus negation), so paraphrases with no shared
  words are not merged.
- Quotes are picked by keyword cues. When no short, safe message matches, the table shows `—`.
- The whole corpus is loaded into memory for each step; very large histories (hundreds of
  thousands of chats) will be slow.

## License

MIT
