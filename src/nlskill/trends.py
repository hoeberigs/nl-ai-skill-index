"""Track how the composition of demand shifts over time.

Counting live vacancies per week would confound two different things: how many
roles employers are opening, and how long each one lingers. A cluster can hold
a large share of the board simply because its vacancies are hard to fill. So
inflow (vacancies first seen in a week) and stock (vacancies live in a week)
are reported separately, and the difference between them is informative.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict

from .data import Panel, Spell


def _week_key(d: dt.date) -> str:
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def weekly_series(
    panel: Panel, spells: list[Spell], group_of: dict[str, str], min_weeks: int = 3
) -> dict:
    """Weekly inflow and stock per group."""
    dates = panel.run_dates
    weeks: list[str] = []
    seen_weeks: set[str] = set()
    for d in dates:
        w = _week_key(d)
        if w not in seen_weeks:
            seen_weeks.add(w)
            weeks.append(w)
    widx = {w: i for i, w in enumerate(weeks)}

    groups = sorted(set(group_of.values()))
    inflow = {g: [0] * len(weeks) for g in groups}
    stock = {g: [0] * len(weeks) for g in groups}

    for s in spells:
        g = group_of.get(s.key)
        if g is None:
            continue
        wi = widx[_week_key(dates[s.start_idx])]
        inflow[g][wi] += 1
        # A vacancy counts towards the stock of every week it was live in.
        touched = {
            _week_key(dates[i])
            for i in range(s.start_idx, min(s.last_seen_idx, len(dates) - 1) + 1)
        }
        for w in touched:
            stock[g][widx[w]] += 1

    # The first and last weeks are usually partial, which produces a fake dip
    # at both ends of every line.
    complete = [
        i for i, w in enumerate(weeks) if sum(1 for d in dates if _week_key(d) == w) >= 5
    ]
    lo, hi = (complete[0], complete[-1]) if complete else (0, len(weeks) - 1)

    weeks_t = weeks[lo : hi + 1]
    out_inflow = {g: v[lo : hi + 1] for g, v in inflow.items()}
    out_stock = {g: v[lo : hi + 1] for g, v in stock.items()}

    shares = []
    for i in range(len(weeks_t)):
        total = sum(out_inflow[g][i] for g in groups) or 1
        shares.append({g: round(out_inflow[g][i] / total, 4) for g in groups})

    return {
        "weeks": weeks_t,
        "inflow": out_inflow,
        "stock": out_stock,
        "inflow_share": shares,
        "partial_weeks_trimmed": [weeks[:lo], weeks[hi + 1 :]],
    }


def skill_trends(panel: Panel, spells: list[Spell], top_n: int = 12) -> dict:
    """Share of new vacancies mentioning each skill, by week.

    Expressed as a share rather than a count so that a week with more scraping
    does not look like a week with more demand.
    """
    dates = panel.run_dates
    weeks: list[str] = []
    for d in dates:
        w = _week_key(d)
        if not weeks or weeks[-1] != w:
            weeks.append(w)
    widx = {w: i for i, w in enumerate(weeks)}

    counts: dict[str, list[int]] = defaultdict(lambda: [0] * len(weeks))
    totals = [0] * len(weeks)
    overall: dict[str, int] = defaultdict(int)

    for s in spells:
        wi = widx[_week_key(dates[s.start_idx])]
        totals[wi] += 1
        skills = {x.strip() for x in (s.skills or "").split(",") if x.strip()}
        for sk in skills:
            counts[sk][wi] += 1
            overall[sk] += 1

    top = [k for k, _ in sorted(overall.items(), key=lambda kv: -kv[1])[:top_n]]
    complete = [
        i for i, w in enumerate(weeks) if sum(1 for d in dates if _week_key(d) == w) >= 5
    ]
    lo, hi = (complete[0], complete[-1]) if complete else (0, len(weeks) - 1)

    series = {
        k: [
            round(counts[k][i] / totals[i], 4) if totals[i] else 0.0
            for i in range(lo, hi + 1)
        ]
        for k in top
    }
    return {
        "weeks": weeks[lo : hi + 1],
        "series": series,
        "totals": totals[lo : hi + 1],
        "overall_counts": {k: overall[k] for k in top},
    }


def trend_slope(series: list[float]) -> float:
    """Least-squares slope per week, used to rank risers and fallers."""
    n = len(series)
    if n < 3:
        return 0.0
    xbar = (n - 1) / 2
    ybar = sum(series) / n
    num = sum((i - xbar) * (y - ybar) for i, y in enumerate(series))
    den = sum((i - xbar) ** 2 for i in range(n))
    return num / den if den else 0.0
