# Roadmap

The pre-release review ranked five improvements. The first two (an enforced private
classification mode with an exact `--preview`, and transactional, budgeted processing) are in the
current release. These are next, in order.

## 1. Learn specific, scoped preferences (large)

Today each question maps to fixed guidance: the tool can tell that a user kept correcting
formatting, but not whether they wanted more structure or less. Planned:

- Extract, per correction, the triple *what the agent did -> what the user wanted -> in what
  context* (repository, language, task type) instead of a yes/no score.
- Separate project rules ("this repo uses pnpm") from personal preferences ("no headings in
  short replies"), and route them to the project's `AGENTS.md` / `CLAUDE.md` or the user's global
  file accordingly.
- Detect contradictions (asked for more detail in one project, less in another) and surface them
  for the user to resolve instead of picking a generic template.
- Keep the extraction step behind the same isolation, redaction, budget and preview guarantees
  as tagging.

## 2. Make every recommendation auditable (medium)

- Stable message identities alongside the stable session identities that exist today, so each
  quote links back to one message.
- A local evidence view (`session-skills show PATTERN`) listing every supporting chat with its
  date, redacted excerpt and scores from both models.
- Record the user's accept / reject decision per proposed skill or rule, and stop re-proposing
  rejected ones.
- Per-pattern support requirements that scale with how rare the pattern is, not one global
  minimum.

## 3. Close the improvement loop (large)

Agent-aware installation with diffs and rollback exists now. Next:

- After an install, measure whether the corrections it targeted become rarer in later sessions
  (a before/after rate per pattern, with the sample sizes shown).
- Flag installed skills that never change anything, and offer to roll them back.
- Detect when an agent's instruction files changed outside this tool and refuse to overwrite
  hand edits inside the managed block.

## Smaller items

- Stream the corpus instead of loading it into memory for very large histories.
- Pass the OpenCode prompt on stdin once `opencode run` supports it, so it is not visible in the
  process list.
- Revisit the Codex backend when `codex exec` offers a way to disable every tool.
