"""Recover the vacancies the keyword taxonomy could not place.

The upstream tracker labels roles with keyword rules. That is transparent and
cheap, and it fails whenever an advert describes a role without using the
expected words, which is common in Dutch listings and in adverts written by
recruiters rather than engineers. About a fifth of postings end up as "other".

Those postings are not unlabellable, only unmatched: a model trained on the
listings the rules did place can read the rest. The value is testable, so the
gain is cross-validated and reported rather than asserted, and a confidence
floor keeps genuinely ambiguous adverts in "other" instead of forcing every
one into a class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, f1_score

from .config import CV_FOLDS, RANDOM_STATE, RESCUE_THRESHOLD


@dataclass
class RescueResult:
    macro_f1: float
    accuracy: float
    per_class: dict
    n_train: int
    n_other: int
    n_rescued: int
    rescue_rate: float
    assignments: dict[int, str]
    confidences: dict[int, float]
    rescued_breakdown: dict[str, int]


def rescue_other(
    embeddings: np.ndarray,
    categories: list[str],
    min_class: int = 20,
    threshold: float = RESCUE_THRESHOLD,
) -> RescueResult:
    """Train on confidently-labelled vacancies, then relabel the "other" bucket."""
    cats = np.asarray(categories, dtype=object)
    is_other = np.array([c in ("", "other", "unknown", None) for c in cats])

    # Classes with too few examples cannot be learned or honestly evaluated,
    # so they stay out of the training set rather than adding noisy folds.
    labelled = ~is_other
    counts: dict[str, int] = {}
    for c in cats[labelled]:
        counts[c] = counts.get(c, 0) + 1
    keep_classes = {c for c, n in counts.items() if n >= min_class}
    train_mask = labelled & np.array([c in keep_classes for c in cats])

    X, y = embeddings[train_mask], cats[train_mask].astype(str)
    if len(set(y.tolist())) < 2:
        return RescueResult(0.0, 0.0, {}, int(train_mask.sum()), int(is_other.sum()), 0, 0.0, {}, {}, {})

    clf = LogisticRegression(
        max_iter=2000, C=4.0, class_weight="balanced", random_state=RANDOM_STATE
    )

    # Honest generalisation estimate before the model is used for anything.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=1)
    macro = float(f1_score(y, pred, average="macro"))
    acc = float((pred == y).mean())
    per_class = classification_report(y, pred, output_dict=True, zero_division=0)

    clf.fit(X, y)
    other_idx = np.nonzero(is_other)[0]
    assignments: dict[int, str] = {}
    confidences: dict[int, float] = {}
    breakdown: dict[str, int] = {}
    if other_idx.size:
        proba = clf.predict_proba(embeddings[other_idx])
        best = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        for i, b, c in zip(other_idx, best, conf):
            confidences[int(i)] = float(c)
            if c >= threshold:
                lab = str(clf.classes_[b])
                assignments[int(i)] = lab
                breakdown[lab] = breakdown.get(lab, 0) + 1

    return RescueResult(
        macro_f1=round(macro, 4),
        accuracy=round(acc, 4),
        per_class={
            k: {kk: round(vv, 4) for kk, vv in v.items()}
            for k, v in per_class.items()
            if isinstance(v, dict)
        },
        n_train=int(train_mask.sum()),
        n_other=int(is_other.sum()),
        n_rescued=len(assignments),
        rescue_rate=round(len(assignments) / max(1, int(is_other.sum())), 4),
        assignments=assignments,
        confidences=confidences,
        rescued_breakdown=dict(sorted(breakdown.items(), key=lambda kv: -kv[1])),
    )
