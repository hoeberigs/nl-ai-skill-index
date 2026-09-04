"""Recover true vacancy lifetimes by inverting the observation process.

A grace period repairs gaps inside a sighting history, but it cannot repair
the edges. A vacancy seen exactly once has a measured span of one day however
long it was really live, and a third of this panel is in that position. So the
observed span distribution is not the lifetime distribution; it is the
lifetime distribution pushed through a lossy instrument.

Rather than correcting the estimate after the fact, this module runs the
instrument forwards. It assumes a parametric lifetime distribution, simulates
vacancies passing through the measured detection curve and the same spell
splitting the real pipeline applies, and searches for the parameters whose
simulated output matches the observed summary statistics. That is simulated
method of moments, and it makes the observation model an explicit, testable
object instead of a caveat in a footnote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from .detection import DetectionCurve, MarkovDetection


@dataclass
class Calibration:
    mu: float
    sigma: float
    ephemeral_share: float
    median_true: float
    mean_true: float
    p90_true: float
    observed_targets: dict
    simulated_targets: dict
    loss: float
    inflation_factor: float

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def _detection_lookup(curve: DetectionCurve, n_runs: int) -> np.ndarray:
    """Smooth per-age detection probabilities with a support-weighted fallback."""
    p = np.full(n_runs + 1, curve.overall(), dtype=float)
    for age, d, t in zip(curve.ages, curve.detected, curve.trials):
        if age <= n_runs and t >= 30:
            p[age] = d / t
    # Day zero is 1.0 by construction (a vacancy enters the panel by being
    # seen), which is a property of the panel rather than the instrument, so
    # use the day-one rate in its place.
    if n_runs >= 1:
        p[0] = p[1]
    # Forward-fill the sparse tail so old ages inherit the last supported value.
    last = p[0]
    for i in range(len(p)):
        if p[i] <= 0:
            p[i] = last
        last = p[i]
    return p


def _observed_stats(spans: np.ndarray, singles: np.ndarray) -> dict:
    if spans.size == 0:
        return {"single_share": 0.0, "median_span": 0.0, "p90_span": 0.0}
    return {
        "single_share": float(singles.mean()),
        "median_span": float(np.median(spans)),
        "p90_span": float(np.percentile(spans, 90)),
    }


def simulate(
    mu: float,
    sigma: float,
    ephemeral_share: float,
    p_by_age: np.ndarray,
    stay_visible: float,
    n_runs: int,
    grace_days: int,
    n_sim: int,
    rng: np.random.Generator,
) -> dict:
    """Push a candidate lifetime distribution through the measured instrument.

    Detection is simulated as the fitted two-state chain rather than as
    independent draws. Persistence is held at the measured value and the
    probability of surfacing is solved per age so that the chain's stationary
    visibility reproduces the measured marginal detection curve:

        a01(age) = p(age) * (1 - a11) / (1 - p(age))
    """
    # True lifetimes as a two-component mixture. A third of this panel is
    # seen exactly once, which no single heavy-tailed distribution reproduces
    # alongside a 38-day upper decile. The two components correspond to two
    # things the board actually contains: listings that appear briefly and
    # never establish themselves (aggregator duplicates, instantly-filled or
    # withdrawn adverts), and genuine vacancies that run for weeks.
    life = np.maximum(1, np.round(rng.lognormal(mu, sigma, n_sim)).astype(int))
    if ephemeral_share > 0:
        eph = rng.random(n_sim) < ephemeral_share
        life = np.where(eph, 1, life)
    life = np.minimum(life, n_runs * 3)
    # Vacancies are already flowing before observation starts, so entry days
    # extend well before day zero. Without this the panel would be filled with
    # artificially young vacancies.
    entry = rng.integers(-n_runs, n_runs, n_sim)

    spans, singles = [], []
    max_age = len(p_by_age) - 1
    for L, e in zip(life, entry):
        # Days this vacancy is both alive and inside the observation window.
        lo, hi = max(0, e), min(n_runs - 1, e + L - 1)
        if hi < lo:
            continue
        ages = np.arange(lo - e, hi - e + 1)
        probs = np.clip(p_by_age[np.minimum(ages, max_age)], 1e-6, 1 - 1e-6)
        # Solve the surfacing probability that yields this marginal under the
        # measured persistence.
        a01 = np.clip(probs * (1.0 - stay_visible) / (1.0 - probs), 1e-6, 1.0)
        draws = rng.random(ages.size)
        hits = np.empty(ages.size, dtype=bool)
        # Start the chain from its stationary distribution for this age.
        visible = draws[0] < probs[0]
        hits[0] = visible
        for i in range(1, ages.size):
            thresh = stay_visible if visible else a01[i]
            visible = draws[i] < thresh
            hits[i] = visible
        idx = np.nonzero(hits)[0]
        if idx.size == 0:
            continue
        days = idx + lo
        # Left-censored spells are excluded from the real analysis, so exclude
        # them here too or the comparison is not like for like.
        if days[0] == 0:
            continue
        # Apply the same spell splitting as the pipeline.
        blocks, cur = [], [days[0]]
        for a, b in zip(days, days[1:]):
            if b - a - 1 >= grace_days:
                blocks.append(cur)
                cur = [b]
            else:
                cur.append(b)
        blocks.append(cur)
        for blk in blocks:
            spans.append(blk[-1] - blk[0] + 1)
            singles.append(1.0 if len(blk) == 1 else 0.0)

    return _observed_stats(np.asarray(spans, dtype=float), np.asarray(singles, dtype=float))


def _loss(sim: dict, obs: dict) -> float:
    """Relative squared error across the summary statistics being matched."""
    total = 0.0
    for k, w in (("single_share", 2.0), ("median_span", 1.0), ("p90_span", 0.7)):
        o = obs[k]
        if o <= 0:
            continue
        total += w * ((sim[k] - o) / o) ** 2
    return total


def calibrate(
    observed: dict,
    curve: DetectionCurve,
    markov: MarkovDetection,
    n_runs: int,
    grace_days: int,
    n_sim: int = 12000,
    seed: int = 42,
) -> Calibration:
    """Grid-search then refine the lifetime parameters that reproduce the data."""
    rng = np.random.default_rng(seed)
    p_by_age = _detection_lookup(curve, n_runs)

    best = None
    # Coarse pass over a wide, deliberately generous parameter range.
    for w in np.arange(0.0, 0.86, 0.1):
        for mu in np.arange(1.6, 4.21, 0.3):
            for sigma in np.arange(0.4, 1.61, 0.2):
                sim = simulate(
                    mu, sigma, float(w), p_by_age, markov.p_stay_visible,
                    n_runs, grace_days, n_sim, rng,
                )
                l = _loss(sim, observed)
                if best is None or l < best[0]:
                    best = (l, float(mu), float(sigma), float(w), sim)

    # Local refinement around the coarse optimum.
    _, mu0, sig0, w0, _ = best
    for w in np.arange(max(0.0, w0 - 0.08), w0 + 0.081, 0.04):
        for mu in np.arange(mu0 - 0.3, mu0 + 0.31, 0.1):
            for sigma in np.arange(max(0.2, sig0 - 0.2), sig0 + 0.21, 0.1):
                sim = simulate(
                    mu, sigma, float(w), p_by_age, markov.p_stay_visible,
                    n_runs, grace_days, n_sim * 2, rng,
                )
                l = _loss(sim, observed)
                if l < best[0]:
                    best = (l, float(mu), float(sigma), float(w), sim)

    loss, mu, sigma, w, sim = best
    # These describe the substantive component: genuine vacancies, excluding
    # the ephemeral listings the mixture isolates. Reporting a blended median
    # across both would answer a question nobody asks, since the ephemeral
    # component is mostly an artefact of how boards syndicate adverts.
    median_true = math.exp(mu)
    mean_true = math.exp(mu + sigma**2 / 2)
    p90_true = math.exp(mu + 1.2816 * sigma)
    return Calibration(
        mu=mu,
        sigma=sigma,
        ephemeral_share=w,
        median_true=median_true,
        mean_true=mean_true,
        p90_true=p90_true,
        observed_targets=observed,
        simulated_targets={k: round(v, 4) for k, v in sim.items()},
        loss=loss,
        inflation_factor=median_true / max(observed["median_span"], 1e-9),
    )


def observed_targets(spells) -> dict:
    spans = np.array([s.duration_days for s in spells], dtype=float)
    singles = np.array([1.0 if s.sightings == 1 else 0.0 for s in spells])
    return _observed_stats(spans, singles)


def calibrate_robust(
    observed: dict,
    curve: DetectionCurve,
    markov: MarkovDetection,
    n_runs: int,
    grace_days: int,
    seeds: tuple[int, ...] = (1, 7, 42, 99, 2026),
    n_sim: int = 6000,
) -> dict:
    """Repeat the fit across seeds and report the spread, not a point estimate.

    The fit is a stochastic search over a simulated criterion, so a single run
    reports its own random draw as if it were the answer. Refitting under
    several seeds and publishing the range costs a few minutes and makes the
    precision of the estimate visible instead of implied.
    """
    fits = [
        calibrate(observed, curve, markov, n_runs, grace_days, n_sim=n_sim, seed=s)
        for s in seeds
    ]
    med = sorted(f.median_true for f in fits)
    mean = sorted(f.mean_true for f in fits)
    eph = sorted(f.ephemeral_share for f in fits)
    best = min(fits, key=lambda f: f.loss)
    mid = len(fits) // 2
    return {
        "seeds": list(seeds),
        "median_true_days": round(med[mid], 1),
        "median_true_range": [round(med[0], 1), round(med[-1], 1)],
        "mean_true_days": round(mean[mid], 1),
        "mean_true_range": [round(mean[0], 1), round(mean[-1], 1)],
        "ephemeral_share": round(eph[mid], 3),
        "ephemeral_share_range": [round(eph[0], 3), round(eph[-1], 3)],
        "best_fit": best.to_dict(),
        "worst_loss": round(max(f.loss for f in fits), 5),
    }
