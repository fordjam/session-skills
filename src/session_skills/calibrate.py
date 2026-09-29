"""``calibrate``: check the cheap tagger against a second model from a different family.

A stratified sample (low / middle / high total score) of tagged chats is labelled again by the
second backend with the same redacted text and questions. The sample is split in two by chat id:
the flag threshold is chosen on one half and agreement and Cohen's kappa are measured on the
other, so a question is never judged on the data that tuned it.

Statuses:
  reliable      enough support (``min_support`` chats, ``min_positives`` labeller positives) and
                held-out agreement and kappa above the bar; its tuned threshold is used
  insufficient  too little evidence either way; the default threshold is used, marked unverified
  unreliable    the two models disagree; the question is left out of the brief
  untested      neither model flagged it in the sample

Every label and the report carry provenance (backend, prompt version, payload hash). Labels are
reused only when all three still match, and ``brief`` ignores a calibration whose sample tags or
labels have changed since.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .backends import Backend, FatalBackendError, family_of_spec
from .collect import refresh_redaction
from .config import Config
from .questions import QUESTION_NAMES
from .state import State
from .tagging import Budget, BudgetExhausted, ChatFailed, label_chat, payload_hash, payloads, prompt_version

MIN_KAPPA = 0.4


class CalibrationError(RuntimeError):
    pass


def stratified_sample(tags: dict[str, dict[str, Any]], size: int, seed: int = 0) -> list[str]:
    """Deterministic sample of ``size`` chat ids spread over score tertiles."""
    ranked = sorted(tags, key=lambda cid: (sum(tags[cid]["scores"].values()), cid))
    if len(ranked) <= size:
        return ranked
    rng = random.Random(seed)
    third = len(ranked) // 3
    strata = [ranked[:third], ranked[third : 2 * third], ranked[2 * third :]]
    quotas = [size // 3] * 3
    for i in range(size - sum(quotas)):
        quotas[-(i + 1)] += 1  # extra picks go to the high-score stratum first
    picked: list[str] = []
    for stratum, quota in zip(strata, quotas, strict=True):
        picked += rng.sample(stratum, min(quota, len(stratum)))
    return sorted(picked)


def held_out(cid: str) -> bool:
    """Deterministic half split of the sample by chat id."""
    return hashlib.sha256(cid.encode()).digest()[0] % 2 == 1


def _kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    observed = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


@dataclass
class QuestionCalibration:
    agreement: float
    kappa: float | None
    threshold: float
    n: int
    held_out: int
    tagger_positive: int
    labeller_positive: int
    status: str  # reliable | insufficient | unreliable | untested

    def as_dict(self) -> dict[str, Any]:
        return {
            "agreement": round(self.agreement, 3),
            "kappa": None if self.kappa is None else round(self.kappa, 3) + 0.0,
            "threshold": self.threshold,
            "n": self.n,
            "held_out": self.held_out,
            "tagger_positive": self.tagger_positive,
            "labeller_positive": self.labeller_positive,
            "status": self.status,
        }


def _best_threshold(pairs: list[tuple[float, bool]], default: float) -> float:
    if not pairs:
        return default
    best_t, best_a = default, -1.0
    for step in range(6, 19):  # thresholds 0.30 .. 0.90
        t = round(step * 0.05, 2)
        agree = sum((mine >= t) == lab for mine, lab in pairs) / len(pairs)
        if agree > best_a + 1e-9 or (abs(agree - best_a) < 1e-9 and abs(t - default) < abs(best_t - default)):
            best_t, best_a = t, agree
    return best_t


def score_question(
    pairs: list[tuple[float, float]],
    default: float,
    min_agreement: float,
    *,
    ids: list[str] | None = None,
    min_support: int = 20,
    min_positives: int = 4,
) -> QuestionCalibration:
    """``pairs`` are (tagger probability, labeller probability); ``ids`` the matching chat ids,
    used for the fit / held-out split."""
    if not pairs:
        return QuestionCalibration(0.0, None, default, 0, 0, 0, 0, "untested")
    ids = ids or [f"#{i}" for i in range(len(pairs))]
    labelled = [(mine, other >= 0.5) for mine, other in pairs]
    fit = [p for p, cid in zip(labelled, ids, strict=True) if not held_out(cid)]
    test = [p for p, cid in zip(labelled, ids, strict=True) if held_out(cid)]
    if len(fit) < 2 or len(test) < 2:  # too small to split: judge at the default threshold
        threshold, test = default, labelled
    else:
        threshold = _best_threshold(fit, default)
    flags = [(mine >= threshold, lab) for mine, lab in test]
    agreement = sum(a == b for a, b in flags) / len(flags)
    kappa = _kappa(flags)
    pos_tagger = sum(mine >= threshold for mine, _ in labelled)
    pos_labeller = sum(lab for _, lab in labelled)
    if pos_tagger == 0 and pos_labeller == 0:
        status = "untested"  # nothing to disagree about in the sample
    elif agreement < min_agreement or (kappa is not None and kappa < MIN_KAPPA):
        status = "unreliable"  # evidence of disagreement is enough to exclude, even when small
    elif len(pairs) < min_support or pos_labeller < min_positives or kappa is None:
        status = "insufficient"  # never call a question reliable on thin evidence
    else:
        status = "reliable"
    return QuestionCalibration(agreement, kappa, threshold, len(pairs), len(flags), pos_tagger, pos_labeller, status)


def fingerprint(sample: list[str], tags: dict[str, dict[str, Any]], labels: dict[str, dict[str, Any]]) -> str:
    """What a calibration was computed from. Any change to a sampled tag or label, the prompt or
    the backends changes it."""
    rows = [
        [
            cid,
            *(str(tags.get(cid, {}).get(k, "")) for k in ("backend", "prompt_version", "payload_hash")),
            *(str(labels.get(cid, {}).get(k, "")) for k in ("backend", "prompt_version", "payload_hash")),
        ]
        for cid in sorted(sample)
    ]
    return hashlib.sha256(json.dumps([prompt_version(), rows]).encode()).hexdigest()[:16]


def current_calibration(state: State) -> tuple[dict[str, Any], str]:
    """(calibration report, "" ) when it still describes the current tags and labels, otherwise
    ({}, reason)."""
    report = state.read_json("calibration.json")
    if not report:
        return {}, ""
    sample = [str(c) for c in report.get("sample_ids", [])]
    if not sample or report.get("fingerprint") != fingerprint(sample, state.tags(), state.labels()):
        return {}, "calibration is out of date (tags, transcripts, prompt or backends changed); re-run calibrate"
    return report, ""


def calibrate(
    cfg: Config,
    state: State,
    labeller: Backend,
    *,
    allow_same_family: bool = False,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    state.lock()
    refresh_redaction(cfg, state)
    chats = state.chats()
    tags = {cid: t for cid, t in state.tags().items() if cid in chats and isinstance(t.get("scores"), dict)}
    if not tags:
        raise CalibrationError("no tagged chats yet; run `session-skills tag` first")
    tag_backends = sorted({str(t.get("backend", "")) for t in tags.values()})
    tagger_families = {family_of_spec(spec) for spec in tag_backends if spec}
    if labeller.family() in tagger_families and not allow_same_family:
        raise CalibrationError(
            f"the calibration backend ({labeller.spec}) is the same model family as a tagger "
            f"({', '.join(tag_backends)}); pick a different family or pass --allow-same-family"
        )
    sample = stratified_sample(tags, cfg.sample_size, seed)
    labels = state.labels()
    budget = Budget(max(0, cfg.calibrate_max_requests))
    failed = 0
    stopped = ""
    version = prompt_version()
    for i, cid in enumerate(sample, 1):
        existing = labels.get(cid)
        current_hash = payload_hash(payloads(chats[cid], cfg))
        if (
            existing
            and existing.get("backend") == labeller.spec
            and existing.get("prompt_version") == version
            and existing.get("payload_hash") == current_hash
        ):
            continue
        if budget.remaining == 0:
            stopped = f"request budget of {budget.limit} reached; re-run to continue"
            break
        try:
            answer, provenance = label_chat(chats[cid], labeller, cfg, budget, state, "label")
        except BudgetExhausted:
            stopped = f"request budget of {budget.limit} reached; re-run to continue"
            break
        except FatalBackendError as exc:
            stopped = str(exc)
            break
        except ChatFailed:
            failed += 1
            continue
        labels[cid] = {"chat_id": cid, **provenance, **answer}
        state.save_label(cid, labels[cid])
        if progress:
            progress(f"labelled {i}/{len(sample)} ({budget.used}/{budget.limit} requests)")

    def valid(cid: str) -> bool:
        label = labels.get(cid, {})
        return (
            label.get("backend") == labeller.spec
            and label.get("prompt_version") == version
            and label.get("payload_hash") == payload_hash(payloads(chats[cid], cfg))
        )

    labelled = [cid for cid in sample if valid(cid)]
    questions: dict[str, Any] = {}
    unreliable = []
    for name in QUESTION_NAMES:
        pairs = [(float(tags[cid]["scores"][name]), float(labels[cid]["scores"][name])) for cid in labelled]
        result = score_question(
            pairs,
            cfg.default_threshold,
            cfg.min_agreement,
            ids=labelled,
            min_support=cfg.min_support,
            min_positives=cfg.min_positives,
        )
        questions[name] = result.as_dict()
        if result.status == "unreliable":
            unreliable.append(name)
    report = {
        "sample": len(sample),
        "sample_ids": labelled,
        "labelled": len(labelled),
        "label_failures": failed,
        "requests": budget.used,
        "stopped": stopped,
        "tagger": tag_backends,
        "labeller": labeller.spec,
        "labeller_family": labeller.family(),
        "min_agreement": cfg.min_agreement,
        "min_support": cfg.min_support,
        "min_positives": cfg.min_positives,
        "prompt_version": version,
        "fingerprint": fingerprint(labelled, tags, labels),
        "questions": questions,
        "unreliable": unreliable,
    }
    state.write_json("calibration.json", report)
    return report
