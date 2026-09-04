"""Kaplan-Meier estimation of how long a vacancy stays advertised.

Right-censoring matters here: a vacancy still live on the final snapshot has
not told us its duration, only a lower bound on it. Dropping those, or
treating the last snapshot as an ending, both understate tenure. Kaplan-Meier
uses the information they do carry without inventing the part they do not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data import Spell


@dataclass
class KMCurve:
    times: list[int]
    survival: list[float]
    lower: list[float]
    upper: list[float]
    at_risk: list[int]
    events: list[int]
    n: int
    n_events: int
    n_censored: int

    def quantile(self, q: float) -> float | None:
        """Smallest duration at which survival drops to or below 1 - q."""
        target = 1.0 - q
        for t, s in zip(self.times, self.survival):
            if s <= target:
                return float(t)
        return None

    @property
    def median(self) -> float | None:
        return self.quantile(0.5)

    def restricted_mean(self, horizon: int) -> float:
        """Mean days advertised within `horizon`, the area under the curve.

        Reported because a median is undefined whenever the curve never falls
        below 0.5 within the observation window, which happens for the
        longest-lived role clusters.
        """
        area, prev_t, prev_s = 0.0, 0, 1.0
        for t, s in zip(self.times, self.survival):
            if t > horizon:
                break
            area += prev_s * (t - prev_t)
            prev_t, prev_s = t, s
        area += prev_s * max(0, horizon - prev_t)
        return area

    def as_dict(self) -> dict:
        return {
            "times": self.times,
            "survival": [round(s, 5) for s in self.survival],
            "lower": [round(s, 5) for s in self.lower],
            "upper": [round(s, 5) for s in self.upper],
            "n": self.n,
            "n_events": self.n_events,
            "n_censored": self.n_censored,
            "median": self.median,
            "p25": self.quantile(0.25),
            "p75": self.quantile(0.75),
        }


def kaplan_meier(spells: list[Spell], z: float = 1.96) -> KMCurve:
    """Product-limit estimator with Greenwood confidence bands."""
    obs = [(s.duration_days, s.observed_event) for s in spells]
    if not obs:
        return KMCurve([], [], [], [], [], [], 0, 0, 0)

    obs.sort(key=lambda x: x[0])
    n = len(obs)
    distinct = sorted({t for t, _ in obs})

    times, surv, lo, hi, risk_out, ev_out = [], [], [], [], [], []
    s = 1.0
    # Greenwood's formula accumulates the variance of log S(t).
    var_acc = 0.0
    at_risk = n

    for t in distinct:
        d = sum(1 for tt, e in obs if tt == t and e)
        censored = sum(1 for tt, e in obs if tt == t and not e)
        if at_risk <= 0:
            break
        if d:
            s *= 1.0 - d / at_risk
            if at_risk - d > 0:
                var_acc += d / (at_risk * (at_risk - d))
        se = math.sqrt(var_acc) if var_acc > 0 else 0.0
        # Log-log transform keeps the band inside [0, 1].
        if 0.0 < s < 1.0 and se > 0:
            ll = math.log(-math.log(s))
            half = z * se / abs(math.log(s))
            band = (
                math.exp(-math.exp(ll + half)),
                math.exp(-math.exp(ll - half)),
            )
            l_, u_ = min(band), max(band)
        else:
            l_ = u_ = s

        times.append(t)
        surv.append(s)
        lo.append(max(0.0, l_))
        hi.append(min(1.0, u_))
        risk_out.append(at_risk)
        ev_out.append(d)
        at_risk -= d + censored

    return KMCurve(
        times=times,
        survival=surv,
        lower=lo,
        upper=hi,
        at_risk=risk_out,
        events=ev_out,
        n=n,
        n_events=sum(1 for _, e in obs if e),
        n_censored=sum(1 for _, e in obs if not e),
    )


def logrank(a: list[Spell], b: list[Spell]) -> dict:
    """Two-sample log-rank test, so cluster differences are not read off eyeballs."""
    obs = [(s.duration_days, s.observed_event, 0) for s in a] + [
        (s.duration_days, s.observed_event, 1) for s in b
    ]
    if not obs:
        return {"chi2": 0.0, "p": 1.0}
    times = sorted({t for t, e, _ in obs if e})
    o1 = e1 = v = 0.0
    for t in times:
        n1 = sum(1 for tt, _, g in obs if tt >= t and g == 1)
        n0 = sum(1 for tt, _, g in obs if tt >= t and g == 0)
        nt = n0 + n1
        d1 = sum(1 for tt, ee, g in obs if tt == t and ee and g == 1)
        dt_ = sum(1 for tt, ee, _ in obs if tt == t and ee)
        if nt < 2 or dt_ == 0:
            continue
        o1 += d1
        e1 += dt_ * n1 / nt
        v += dt_ * (n1 / nt) * (1 - n1 / nt) * (nt - dt_) / (nt - 1)
    if v <= 0:
        return {"chi2": 0.0, "p": 1.0}
    chi2 = (o1 - e1) ** 2 / v
    # Survival function of chi-square with one degree of freedom.
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"chi2": round(chi2, 3), "p": round(p, 6)}
