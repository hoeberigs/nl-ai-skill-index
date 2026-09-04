"""Model how often the scraper sees a posting that is definitely still live.

The tracker captures a capped, recency-ranked slice of each board, so a
posting can vanish from view without being withdrawn. Detection is therefore
measurable without external ground truth: between two sightings of the same
vacancy the vacancy was necessarily live throughout, so every day in between
is a trial the scraper either passed or failed.

Two properties of that instrument matter, and the second is easy to miss.
Detection is incomplete, at roughly half of live days. It is also strongly
persistent: a posting sits in the visible top-N for a run of days and then
sits buried for a run of days, rather than flickering independently. Treating
detection as independent understates how long a live posting can stay
invisible by more than an order of magnitude, which in turn sets the grace
period far too short.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import Panel


@dataclass
class MarkovDetection:
    """Two-state visible/buried chain fitted to the sighting histories."""

    p_stay_visible: float  # P(seen tomorrow | seen today)
    p_become_visible: float  # P(seen tomorrow | missed today)
    n_transitions: int

    @property
    def stationary(self) -> float:
        """Long-run share of live days on which the posting is visible."""
        denom = self.p_become_visible + (1.0 - self.p_stay_visible)
        return self.p_become_visible / denom if denom else 0.0

    @property
    def odds_ratio(self) -> float:
        a = self.p_stay_visible
        b = self.p_become_visible
        if not (0 < a < 1 and 0 < b < 1):
            return float("inf")
        return (a / (1 - a)) / (b / (1 - b))

    def false_delisting_rate(self, grace_days: int) -> float:
        """Chance a live posting stays buried for `grace_days` consecutive runs.

        This is the quantity a grace period trades against. Under an
        independence assumption it would be (1 - p) ** K, which is far too
        optimistic once runs of invisibility are accounted for.
        """
        return (1.0 - self.p_become_visible) ** grace_days

    def naive_false_delisting_rate(self, grace_days: int, marginal: float) -> float:
        """The same quantity under the independence assumption, for contrast."""
        return (1.0 - marginal) ** grace_days


@dataclass
class DetectionCurve:
    """Marginal detection probability as a function of vacancy age, in days."""

    ages: list[int]
    detected: list[int]
    trials: list[int]

    @property
    def rates(self) -> list[float]:
        return [d / t if t else 0.0 for d, t in zip(self.detected, self.trials)]

    def overall(self) -> float:
        d, t = sum(self.detected), sum(self.trials)
        return d / t if t else 0.0

    def worst_case(self, min_trials: int = 40) -> float:
        vals = [
            d / t for d, t in zip(self.detected, self.trials) if t >= min_trials and d > 0
        ]
        return min(vals) if vals else self.overall()

    def binned(self, edges: tuple[int, ...] = (0, 1, 7, 14, 30, 60, 90, 10**6)) -> list[dict]:
        out = []
        for lo, hi in zip(edges, edges[1:]):
            d = sum(x for a, x in zip(self.ages, self.detected) if lo <= a < hi)
            t = sum(x for a, x in zip(self.ages, self.trials) if lo <= a < hi)
            if not t:
                continue
            top = "+" if hi > 10**5 else str(hi - 1)
            out.append(
                {
                    "label": f"{lo}" if lo == hi - 1 else f"{lo}-{top}",
                    "age_from": lo,
                    "detected": d,
                    "trials": t,
                    "rate": round(d / t, 4),
                }
            )
        return out


def estimate_detection(panel: Panel, min_span: int = 3) -> DetectionCurve:
    """Marginal detection against age, conditioning on provable presence.

    Day zero is excluded from interpretation because a vacancy enters the
    panel by being seen, making its rate 1.0 by construction rather than by
    measurement.
    """
    max_age = panel.n_runs
    detected = [0] * (max_age + 1)
    trials = [0] * (max_age + 1)

    for seen in panel.histories.values():
        if len(seen) < 2:
            continue
        lo, hi = seen[0], seen[-1]
        if hi - lo < min_span:
            continue
        present = set(seen)
        for t in range(lo, hi + 1):
            age = t - lo
            trials[age] += 1
            if t in present:
                detected[age] += 1

    return DetectionCurve(ages=list(range(max_age + 1)), detected=detected, trials=trials)


def estimate_markov(panel: Panel, min_span: int = 4) -> MarkovDetection:
    """Fit the visible/buried transition probabilities on provably live days."""
    ss = sn = hs = hn = 0
    for seen in panel.histories.values():
        if len(seen) < 2:
            continue
        lo, hi = seen[0], seen[-1]
        if hi - lo < min_span:
            continue
        present = set(seen)
        for t in range(lo, hi):
            nxt = (t + 1) in present
            if t in present:
                sn += 1
                ss += nxt
            else:
                hn += 1
                hs += nxt
    return MarkovDetection(
        p_stay_visible=ss / sn if sn else 0.0,
        p_become_visible=hs / hn if hn else 0.0,
        n_transitions=sn + hn,
    )


def grace_period_table(
    curve: DetectionCurve, markov: MarkovDetection, candidates=(1, 3, 7, 14, 21, 28)
) -> list[dict]:
    """What each candidate grace period costs in false delistings.

    Both columns are reported because the gap between them is the finding: an
    independence assumption makes every threshold look safe.
    """
    marginal = curve.overall()
    return [
        {
            "grace_days": k,
            "false_rate_independent": round(
                markov.naive_false_delisting_rate(k, marginal), 6
            ),
            "false_rate_markov": round(markov.false_delisting_rate(k), 6),
        }
        for k in candidates
    ]
