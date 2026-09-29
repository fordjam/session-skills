"""The fixed question set asked of every chat, plus the guidance each pattern turns into."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    name: str
    ask: str  # what the tagging model is asked (probability 0..1)
    kind: str  # "correction", "positive" or "workflow"
    cues: tuple[str, ...]  # words that make a human message a good illustrative quote
    skill: str  # skill directory name
    title: str
    description: str  # skill frontmatter description: what + when
    rule: str  # one line for AGENTS.md / CLAUDE.md
    instead: tuple[str, ...]
    why: str


QUESTIONS: tuple[Question, ...] = (
    Question(
        "too_long",
        "Did the user complain that a reply was too long, verbose or padded?",
        "correction",
        ("too long", "shorter", "verbose", "concise", "tl;dr", "tldr", "wall of text", "brief"),
        "keep-replies-short",
        "Keep replies short",
        "Use when writing any reply to this user. They repeatedly asked for shorter, less padded answers.",
        "Lead with the answer; keep replies short and cut preamble, recaps and filler.",
        (
            "Put the answer or result in the first sentence.",
            "Drop restatements of the question, recaps of what you did step by step and closing offers.",
            "Use a list only when there are three or more parallel items.",
        ),
        "Long replies hid the answer and the user had to ask for a shorter version.",
    ),
    Question(
        "too_short",
        "Did the user ask for more detail because a reply was too short or incomplete?",
        "correction",
        ("more detail", "elaborate", "expand", "incomplete", "too short", "explain more"),
        "give-enough-detail",
        "Give enough detail",
        "Use when explaining a decision, a bug or a plan. The user repeatedly asked for more detail than they got.",
        "When explaining a decision or a failure, include the reason and the evidence, not just the conclusion.",
        (
            "State the conclusion, then the one or two facts that support it.",
            "For fixes, say what was wrong, why, and how you verified the fix.",
        ),
        "Terse answers forced follow-up questions to recover the reasoning.",
    ),
    Question(
        "wrong_answer",
        "Did the user say an answer, fix or piece of code was wrong or did not work?",
        "correction",
        ("wrong", "incorrect", "doesn't work", "does not work", "didn't work", "still broken", "not right", "still failing"),
        "verify-before-answering",
        "Verify before answering",
        "Use before presenting a fix or factual answer. The user often reported that answers or fixes were wrong.",
        "Before presenting a fix, reproduce the problem and run the check that proves the fix.",
        (
            "Reproduce the failure first, then change code, then re-run the same check.",
            "If you cannot verify, say exactly what is unverified.",
        ),
        "Unverified fixes came back broken and cost another round trip.",
    ),
    Question(
        "ignored_instruction",
        "Did the user point out that the assistant ignored or did not follow an explicit instruction?",
        "correction",
        ("i said", "i asked", "i told you", "ignored", "didn't follow", "not what i asked", "instructions"),
        "follow-explicit-instructions",
        "Follow explicit instructions",
        "Use at the start of every task and before finishing. The user repeatedly had to point out ignored instructions.",
        "Re-read the user's explicit instructions before finishing and check each one is met.",
        (
            "Turn every explicit constraint in the request into a checklist item.",
            "Before replying, confirm each item; if one cannot be met, say so up front.",
        ),
        "Missed constraints meant the work had to be redone.",
    ),
    Question(
        "format",
        "Did the user correct the formatting or structure of a reply (headings, tables, bullets, code blocks)?",
        "correction",
        ("format", "table", "bullet", "markdown", "heading", "code block", "plain text"),
        "match-requested-format",
        "Match the requested format",
        "Use when choosing how to lay out a reply or file. The user repeatedly corrected formatting.",
        "Use the format the user asked for; default to plain prose with minimal markdown.",
        (
            "Mirror any format the user specified (table, list, plain text).",
            "Do not add headings to short replies.",
        ),
        "Format corrections were a recurring source of rework.",
    ),
    Question(
        "tone",
        "Did the user object to the tone of a reply (too casual, too formal, sycophantic, preachy)?",
        "correction",
        ("tone", "sycophantic", "preachy", "patronizing", "condescending", "too formal", "too casual"),
        "keep-a-plain-tone",
        "Keep a plain tone",
        "Use when writing any reply. The user objected to tone more than once.",
        "Write in a plain, direct, neutral tone; no flattery, no lecturing.",
        ("Skip compliments and apologies.", "State facts and recommendations directly."),
        "Tone mismatches distracted from the content.",
    ),
    Question(
        "hedging",
        "Did the user complain about hedging, excessive caveats or refusing to commit to an answer?",
        "correction",
        ("hedg", "caveat", "just tell me", "commit to", "which one", "pick one", "be decisive"),
        "commit-to-an-answer",
        "Commit to an answer",
        "Use when the user asks for a recommendation or a choice. They pushed back on hedged answers.",
        "When asked to choose, pick one option and give the main reason; mention caveats only if they change the choice.",
        ("Give one recommendation first.", "Limit caveats to the ones that would change the decision."),
        "Hedged answers left the decision with the user.",
    ),
    Question(
        "made_up",
        "Did the assistant invent facts, APIs, files, flags or sources that do not exist (per the user)?",
        "correction",
        ("made up", "doesn't exist", "does not exist", "hallucinat", "invented", "no such", "fabricat"),
        "check-that-it-exists",
        "Check that it exists",
        "Use before naming an API, flag, file, package or source. The user caught invented ones.",
        "Before citing an API, flag, file or source, confirm it exists (read the file, run --help, open the doc).",
        (
            "Look it up in the codebase or docs before naming it.",
            "If you cannot confirm it, say it is unconfirmed.",
        ),
        "Invented details cost time to track down and eroded trust.",
    ),
    Question(
        "asked_instead",
        "Did the assistant ask the user a question when it should have just done the task?",
        "correction",
        ("just do it", "don't ask", "stop asking", "you decide", "go ahead", "why are you asking"),
        "act-on-clear-requests",
        "Act on clear requests",
        "Use when about to ask the user a clarifying question. They repeatedly wanted the work done instead.",
        "If the request is clear and reversible, do it; ask only when a choice is irreversible or genuinely ambiguous.",
        (
            "Make a reasonable assumption, state it in one line, and proceed.",
            "Batch any truly necessary questions into one message.",
        ),
        "Unneeded questions stalled work that could have finished.",
    ),
    Question(
        "repeated_myself",
        "Did the user have to repeat an instruction or information they had already given?",
        "correction",
        ("again", "already said", "already told", "as i said", "repeat", "for the third time", "once more"),
        "remember-what-was-said",
        "Remember what was said",
        "Use in long sessions and after context compaction. The user repeatedly had to restate things.",
        "Keep a running list of the user's stated constraints and re-check it before each step.",
        (
            "When a constraint is stated, restate it once in your plan and keep it.",
            "After compaction or a long tool run, re-read the constraints before continuing.",
        ),
        "Repeating instructions is the clearest sign a rule belongs in standing instructions.",
    ),
    Question(
        "praise",
        "Did the user explicitly praise a reply or result?",
        "positive",
        ("great", "perfect", "exactly", "nice", "love it", "well done", "thank"),
        "keep-doing-what-worked",
        "Keep doing what worked",
        "Reference of approaches the user praised.",
        "Keep doing what the user praised (see examples).",
        ("Reuse the approaches in the examples.",),
        "Praised replies show the target style.",
    ),
    Question(
        "claimed_without_checking",
        "Did the assistant claim something was done, fixed or working without evidence (per the user)?",
        "correction",
        ("did you check", "did you test", "did you run", "is it actually", "prove", "evidence", "you said it was"),
        "show-evidence-for-done",
        "Show evidence for done",
        "Use before saying a task is done, fixed or working. The user caught unverified claims.",
        "Never say done, fixed or working without the command and output that shows it.",
        (
            "Quote the test, command or check you ran and its result.",
            "If you did not run a check, say 'not verified' instead of 'done'.",
        ),
        "Claims without evidence turned out wrong and had to be re-checked by the user.",
    ),
    Question(
        "acted_without_approval",
        "Did the assistant take an action the user had not approved (and the user objected)?",
        "correction",
        ("without asking", "didn't ask", "who told you", "i didn't approve", "why did you", "revert", "undo"),
        "ask-before-irreversible-actions",
        "Ask before irreversible actions",
        "Use before pushing, deleting, deploying, sending or changing config. The user objected to unapproved actions.",
        "Ask before anything irreversible or outside the request (push, delete, deploy, send, config changes).",
        (
            "List the action and its effect and wait for a yes.",
            "Stay inside the scope of the request; propose extras instead of doing them.",
        ),
        "Unapproved actions had to be undone.",
    ),
    Question(
        "status_request",
        "Did the user have to ask for status or progress (what is running, what is done, what is next)?",
        "workflow",
        ("status", "progress", "where are we", "what's left", "is it done", "still running", "update?"),
        "report-status-proactively",
        "Report status proactively",
        "Use during multi-step or long-running work. The user repeatedly had to ask for status.",
        "During long work, report status unprompted: done, in progress, blocked, next.",
        (
            "After each milestone, post a two-line status: done / next.",
            "When blocked or waiting, say so immediately.",
        ),
        "Status requests are interruptions that a routine update would prevent.",
    ),
    Question(
        "approval_friction",
        "Did the user have to type approvals or confirmations repeatedly just to unblock routine work?",
        "workflow",
        ("yes", "approve", "go ahead", "proceed", "continue", "ok do it", "confirmed"),
        "batch-approvals",
        "Batch approvals",
        "Use when work will need several confirmations. The user spent many turns just approving.",
        "Group routine confirmations into one upfront plan the user can approve once.",
        (
            "Propose the full plan with every step that needs approval, then ask once.",
            "Do not ask to continue after each routine step.",
        ),
        "Repeated approval turns are pure overhead and good automation candidates.",
    ),
    Question(
        "jargon",
        "Did the user ask for plainer language or complain about jargon or unexplained terms?",
        "correction",
        ("jargon", "plain english", "plain language", "what does that mean", "eli5", "simpler terms", "acronym"),
        "use-plain-language",
        "Use plain language",
        "Use when explaining anything to this user. They asked for plainer language more than once.",
        "Use plain words; define any unavoidable term or acronym the first time.",
        ("Replace internal names and acronyms with what they do.", "Explain effects, not mechanisms, unless asked."),
        "Jargon made answers harder to act on.",
    ),
)

QUESTION_NAMES: tuple[str, ...] = tuple(q.name for q in QUESTIONS)
BY_NAME = {q.name: q for q in QUESTIONS}

TASKS = ("coding", "debugging", "ops", "research", "writing", "planning", "data", "other")
OUTCOMES = ("resolved", "partly_resolved", "unresolved", "unclear")
