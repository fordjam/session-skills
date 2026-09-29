from __future__ import annotations

import pytest

from session_skills.redact import EMAIL, ENV, KEY, NUMBER, RedactOptions, redact, short_quote


def test_known_tokens_are_removed(secrets):
    for value in (secrets.anthropic_key, secrets.github_token, secrets.aws_key, secrets.slack_token, secrets.jwt):
        out = redact(f"here is my token {value} please use it")
        assert value not in out
        assert KEY in out


def test_bearer_generic_and_hex(secrets):
    out = redact(f"Authorization: Bearer {secrets.bearer_value}\nuse {secrets.generic} and {secrets.hex40}")
    for value in (secrets.bearer_value, secrets.generic, secrets.hex40):
        assert value not in out


def test_email_numbers_money(secrets):
    text = (
        f"mail {secrets.email} about card {secrets.card} / {secrets.card_spaced}, "
        f"account {secrets.account}, rent was $1,234.56 and bonus $3k"
    )
    out = redact(text)
    assert secrets.email not in out and EMAIL in out
    assert secrets.card not in out and secrets.card_spaced not in out and NUMBER in out
    assert secrets.account not in out
    assert "1,234" not in out and "$N" in out and "$3k" not in out


def test_masked_last_four_and_short_numbers_survive():
    out = redact("card ending ****1234, issue #4521, version 3.12.1, date 2026-09-29")
    assert "****1234" in out and "#4521" in out and "3.12.1" in out and "2026-09-29" in out


def test_home_paths(secrets):
    windows = "C:\\Users\\" + "carol" + "\\repo"
    out = redact(f"open {secrets.home_mac}/main.py and {secrets.home_linux} and {windows}")
    assert secrets.home_user not in out and "bob" not in out and "carol" not in out
    assert "~/projects/app/main.py" in out and "~/src/tool" in out


def test_url_path_is_not_a_home_path():
    assert redact("see https://example.org/home/docs") == "see https://example.org/home/docs"


def test_env_contents(secrets):
    fenced = f"```env\nDB_PASSWORD={secrets.password}\nPORT=8080\n```"
    out = redact(fenced)
    assert secrets.password not in out and ENV in out and "8080" not in out

    dotenv = f"my .env has\nFOO=bar-value\nBAZ={secrets.password}"
    out = redact(dotenv)
    assert "bar-value" not in out and secrets.password not in out

    inline = f"export OPENAI_API_KEY={secrets.password} then run; password: {secrets.password}"
    out = redact(inline)
    assert secrets.password not in out

    js = '{"api_key": "' + secrets.password + '", "name": "demo"}'
    out = redact(js)
    assert secrets.password not in out and '"name": "demo"' in out


def test_pasted_blocks():
    assert redact("<pasted_content>anything at all</pasted_content> thanks") == "[PASTED BLOCK] thanks"
    long_fence = "look:\n```\n" + "\n".join(f"line {i}" for i in range(20)) + "\n```\nok?"
    out = redact(long_fence)
    assert "line 7" not in out and "[PASTED BLOCK: 20 lines]" in out and out.endswith("ok?")
    log = "\n".join(f"ERROR at frame {i}" for i in range(15))
    out = redact("stack:\n\n" + log + "\n\nwhat now?")
    assert "frame 3" not in out and "[PASTED BLOCK: 15 lines]" in out and "what now?" in out
    short = "```\nprint('hi')\n```"
    assert redact(short) == short


def test_private_key_and_url_credentials(secrets):
    out = redact(f"key:\n{secrets.pem}\nand postgres://admin:{secrets.password}@db.local/app?token=abc123&x=1")
    assert "MIIEow" not in out and secrets.password not in out and "abc123" not in out
    assert "x=1" in out


def test_ordinary_text_is_untouched():
    samples = [
        "please run the tests in src/session_skills/tagging.py",
        "checkout feature/add-dry-run-flag-2026 and rebase",
        "commit abc1234 broke the build",
        "that's way too long, just show me the diff",
    ]
    for sample in samples:
        assert redact(sample) == sample


def test_deny_words_and_extra_patterns():
    opts = RedactOptions.build(deny_words=["Globex"], extra_patterns=[r"TICKET-\d+"])
    out = redact("Globex asked about TICKET-991 at globex", opts)
    assert "Globex" not in out and "globex" not in out and "TICKET-991" not in out


def test_idempotent(secrets):
    text = " ".join(secrets.all_values()) + f" {secrets.home_mac} $12.00"
    once = redact(text)
    assert redact(once) == once
    for value in secrets.all_values():
        assert value not in once


@pytest.mark.parametrize("value", ["", None])
def test_empty(value):
    assert redact(value) == ""


def test_short_quote_is_single_line_and_capped(secrets):
    quote = short_quote(f"first line\nsecond | line with {secrets.email} " + "x" * 300, 60)
    assert "\n" not in quote and "|" not in quote and len(quote) <= 60 and secrets.email not in quote


# -- credential formats the pre-release review found leaking ------------------------------------

def test_python_dict_and_other_quoted_secret_keys(secrets):
    for text in (
        f"{{'password': '{secrets.password}'}}",  # reviewer's example
        f'{{"db_password": "{secrets.password}"}}',
        f"config['api_key'] = '{secrets.password}'",
        f'settings = {{"client_secret" => "{secrets.password}"}}',
        f"db_password = {secrets.password}",
        f"passphrase: {secrets.password}",
    ):
        out = redact(text)
        assert secrets.password not in out, text
        assert redact(out) == out


def test_cookie_and_authorization_headers(secrets):
    for text in (
        "Cookie: sessionid=" + secrets.session_cookie,  # reviewer's example
        "Set-Cookie: sid=" + secrets.session_cookie + "; HttpOnly; Path=/",
        "curl -H 'Authorization: Token " + secrets.session_cookie + "' https://api.example.org",
        "Proxy-Authorization: Basic " + secrets.session_cookie,
        'curl -H "X-Api-Key: ' + secrets.session_cookie + '" https://api.example.org',
        "sessionid=" + secrets.session_cookie,
    ):
        out = redact(text)
        assert secrets.session_cookie not in out, text
        assert redact(out) == out


def test_private_keys_of_every_kind(secrets):
    for text in (secrets.pem, secrets.pgp, secrets.putty):
        out = redact(f"here:\n{text}\nthanks")
        assert "MIIEow" not in out and "lQOYBF" not in out and "secretbody" not in out
        assert "[REDACTED:private-key]" in out and out.endswith("thanks")


def test_jwts_including_unsigned(secrets):
    for value in (secrets.jwt, secrets.jwt_unsigned):
        out = redact(f"token {value} end")
        assert "eyJzdWIi" not in out and out.endswith(" end")


def test_connection_strings(secrets):
    for text in (
        f"postgres://admin:{secrets.password}@db.local/app",
        f"redis://:{secrets.password}@cache.local:6379/0",
        f"Server=db;Database=app;User Id=sa;Password={secrets.password};",
        f"DefaultEndpointsProtocol=https;AccountName=demo;AccountKey={secrets.password}+/=;",
        f"jdbc:mysql://db.local/app?user=app&password={secrets.password}",
        f"Driver={{SQL Server}};Uid=app;Pwd={secrets.password};",
    ):
        out = redact(text)
        assert secrets.password not in out, text


def test_new_patterns_leave_ordinary_text_alone():
    for sample in (
        "set max_tokens=100 and keep the token budget small",
        "passwords are hard; the cookie jar is empty",
        "the authorization flow is described in docs/auth.md",
    ):
        assert redact(sample) == sample
