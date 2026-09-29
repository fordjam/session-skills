"""``calibrate``: check the cheap tagger against a second model from a different family.

A stratified sample (low / middle / high total score) of tagged chats is labelled again by the
second backend with the same redacted text and questions. For each question we pick the flag
threshold that maximises agreement with the second backend (ties go to the default), and mark the
question unreliable when agreement or Cohen's kappa is too low. Unreliable questions are left out
of the brief.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from .backends import Backend
from .config import Config
from .questions import QUESTION_NAMES
from .state import State
from .tagging import AnswerError, label_chat

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
    for stratum, quota in zip(strata, quotas):
        picked += rng.sample(stratum, min(quota, len(stratum)))
    return sorted(picked)


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
    tagger_positive: int
    labeller_positive: int
    status: str  # reliable | unreliable | untested

    def as_dict(self) -> dict[str, Any]:
        return {
            "agreement": round(self.agreement, 3),
            "kappa": None if self.kappa is None else round(self.kappa, 3) + 0.0,
            "threshold": self.threshold,
            "n": self.n,
            "tagger_positive": self.tagger_positive,
            "labeller_positive": self.labeller_positive,
            "status": self.status,
        }


def score_question(
    pairs: list[tuple[float, float]], default: float, min_agreement: float
) -> QuestionCalibration:
    """``pairs`` are (tagger probability, labeller probability)."""
    if not pairs:
        return QuestionCalibration(0.0, None, default, 0, 0, 0, "untested")
    labels = [other >= 0.5 for _, other in pairs]
    best_t, best_a = default, -1.0
    for step in range(6, 19):  # thresholds 0.30 .. 0.90
        t = round(step * 0.05, 2)
        agree = sum((mine >= t) == lab for (mine, _), lab in zip(pairs, labels)) / len(pairs)
        if agree > best_a + 1e-9 or (abs(agree - best_a) < 1e-9 and abs(t - default) < abs(best_t - default)):
            best_t, best_a = t, agree
    flags = [mine >= best_t for mine, _ in pairs]
    kappa = _kappa(list(zip(flags, labels)))
    pos_tagger, pos_labeller = sum(flags), sum(labels)
    if pos_tagger == 0 and pos_labeller == 0:
        status = "untested"  # nothing to disagree about in the sample
    elif best_a < min_agreement or (kappa is not None and kappa < MIN_KAPPA):
        status = "unreliable"
    else:
        status = "reliable"
    return QuestionCalibration(best_a, kappa, best_t, len(pairs), pos_tagger, pos_labeller, status)


def calibrate(
    cfg: Config,
    state: State,
    labeller: Backend,
    *,
    tagger_family: str | None = None,
    allow_same_family: bool = False,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    tags = state.tags()
    chats = state.chats()
    tags = {cid: t for cid, t in tags.items() if cid in chats and isinstance(t.get("scores"), dict)}
    if not tags:
        raise CalibrationError("no tagged chats yet; run `session-skills tag` first")
    tag_backends = sorted({str(t.get("backend", "")) for t in tags.values()})
    if tagger_family and labeller.family() == tagger_family and not allow_same_family:
        raise CalibrationError(
            f"the calibration backend ({labeller.spec}) is the same model family as the tagger "
            f"({tagger_family}); pick a different family or pass --allow-same-family"
        )
    sample = stratified_sample(tags, cfg.sample_size, seed)
    labels = state.labels()
    failed = 0
    for i, cid in enumerate(sample, 1):
        existing = labels.get(cid)
        if existing and existing.get("backend") == labeller.spec:
            continue
        try:
            answer, _ = label_chat(chats[cid], labeller, cfg)
        except AnswerError:
            failed += 1
            continue
        labels[cid] = {"chat_id": cid, "backend": labeller.spec, **answer}
        state.save_labels(labels)
        if progress:
            progress(f"labelled {i}/{len(sample)}")
    questions: dict[str, Any] = {}
    unreliable = []
    for name in QUESTION_NAMES:
        pairs = [
            (float(tags[cid]["scores"][name]), float(labels[cid]["scores"][name]))
            for cid in sample
            if cid in labels and labels[cid].get("backend") == labeller.spec
        ]
        result = score_question(pairs, cfg.default_threshold, cfg.min_agreement)
        questions[name] = result.as_dict()
        if result.status == "unreliable":
            unreliable.append(name)
    report = {
        "sample": len(sample),
        "labelled": sum(1 for cid in sample if cid in labels and labels[cid].get("backend") == labeller.spec),
        "label_failures": failed,
        "tagger": tag_backends,
        "labeller": labeller.spec,
        "labeller_family": labeller.family(),
        "min_agreement": cfg.min_agreement,
        "questions": questions,
        "unreliable": unreliable,
    }
    state.write_json("calibration.json", report)
    return report
