# Contributing

Thanks for helping. This tool reads people's private agent transcripts, so the bar for privacy is
higher than usual.

## Ground rules

- **Never commit real session data.** Tests and examples use synthetic transcripts only. The
  writers in `src/session_skills/synthetic.py` produce every supported on-disk format.
- **Never put a literal secret in the repo, not even a fake one.** Build secret-shaped test values
  at runtime by concatenation (see `tests/conftest.py`), so secret scanners stay quiet and nobody
  copies a real-looking key.
- **Redaction runs before anything is stored or sent.** New code that reads session text must go
  through `build-list` (which redacts) or call `redact()` itself.
- **Tests stay offline.** Backends are exercised with `FakeBackend` or small fake CLIs written
  into a temp directory. No test may reach the network or a real model.
- **No new runtime dependencies** without discussion. The standard library covers it so far.

## Development

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest
mypy
```

## Common changes

- **A new source.** Add `sources/<name>.py` with an adapter exposing `name` and
  `iter_sessions()`. Keep only human-typed text, clip assistant replies with `clip_assistant`,
  and finish each session with `finish()`. Register it in `sources/__init__.py`, add a writer to
  `synthetic.py`, and add tests covering tool calls, injected context and sub-agents.
- **A new backend.** Add a branch to `make_backend()` in `backends.py`. Send the prompt on stdin
  where the CLI allows it, run in the temporary working directory `CommandBackend` provides, and
  report the model family correctly so `calibrate` can insist on a different one.
- **A new question.** Add a `Question` to `questions.py` with cues, a skill name and guidance. The
  tagging prompt, calibration, report and skills pick it up automatically.
- **Redactor changes.** Add a test with a synthetic value for every new pattern, plus a test that
  ordinary text (paths, branch names, short hashes) is left alone.

## Pull requests

Keep them small and focused, include tests, and describe any change to what leaves the machine
(prompt contents, backend flags) in the PR description.
