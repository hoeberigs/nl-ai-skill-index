"""The demand index itself: what Dutch employers are asking for, and what is moving.

One measurement decision governs this whole module. Skills are read out of
advert text, and the share of adverts whose text was successfully fetched rose
from 20% in March to around 68% by summer. Dividing skill mentions by *all*
vacancies would therefore show almost every skill "rising" through April, which
is the scraper improving rather than the market moving. The denominator here is
always vacancies whose description was actually retrieved, and months whose
sample is too small to carry a share are excluded rather than plotted thin.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

from .data import Panel, Spell

MIN_MONTH_SAMPLE = 60
MIN_MENTION_SAMPLE = 300
MIN_SKILL_TOTAL = 40

# A keyword-driven AI job scraper also catches jobs that merely sit near the
# query terms. In this panel that is mostly staffing-agency production and
# operator work: the largest single employers by advert count were temp
# agencies, not technology companies. Those adverts are real, but they are not
# the market this index is about, and they behave completely differently
# (median one day advertised against roughly two weeks). They are removed by an
# explicit, auditable rule and the removal is counted and reported.
_AI_TERMS = re.compile(
    r"\b("
    r"ai|a\.i\.|artificial intelligence|kunstmatige intelligentie|"
    r"machine learning|deep learning|ml|mlops|llm|genai|generative ai|"
    r"data scien|data engineer|data analy|datascien|"
    r"nlp|natural language|computer vision|neural|"
    r"algoritme|algorithm|model|analytics|statist"
    r")\b",
    re.I,
)
_AI_CATEGORIES = {
    "ml-engineer", "data-scientist", "ai-engineer", "ai-general", "ai-researcher",
    "cv-engineer", "ai-product", "data-engineer", "ai-manager", "data-general",
    "nlp-engineer", "data-analyst",
}


def is_ai_role(s: Spell) -> bool:
    """Whether a vacancy is plausibly an AI, ML or data role.

    Deliberately generous: a vacancy qualifies on its category, its title or a
    named technical skill. The aim is to remove obvious non-AI staffing noise,
    not to police the boundary of the field.
    """
    if (s.category or "").strip().lower() in _AI_CATEGORIES:
        return True
    if _AI_TERMS.search(s.title or ""):
        return True
    if _skills_of(s):
        return True
    return False


def filter_ai(spells: list[Spell]) -> tuple[list[Spell], dict]:
    """Apply the relevance rule and report attrition with its cause."""
    keep = [s for s in spells if is_ai_role(s)]
    dropped = [s for s in spells if not is_ai_role(s)]
    top_dropped: dict[str, int] = defaultdict(int)
    for s in dropped:
        top_dropped[(s.company or "unknown").strip()] += 1
    return keep, {
        "n_in": len(spells),
        "n_kept": len(keep),
        "n_dropped": len(dropped),
        "dropped_share": round(len(dropped) / len(spells), 4) if spells else 0.0,
        "top_dropped_employers": sorted(
            ({"company": c, "n": n} for c, n in top_dropped.items()),
            key=lambda d: -d["n"],
        )[:10],
    }


def described(spells: list[Spell]) -> list[Spell]:
    """The universe for every text-derived statistic."""
    return [s for s in spells if (s.description or "").strip()]


def _skills_of(s: Spell) -> set[str]:
    return {x.strip() for x in (s.skills or "").split(",") if x.strip()}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, which behaves at the small shares seen here.

    A normal-approximation interval on a 3% share with n=200 runs below zero,
    which is both wrong and visibly wrong once drawn.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _norm_sf(z: float) -> float:
    """Two-sided normal tail probability."""
    return math.erfc(abs(z) / math.sqrt(2))


def cochran_armitage(counts: list[int], totals: list[int]) -> dict:
    """Test for a monotone trend in a proportion across ordered periods.

    Preferred over comparing the first month against the last: it uses every
    month, so it is harder to get a result by choosing convenient endpoints,
    and it has more power when a trend is real but gradual.
    """
    n = sum(totals)
    r = sum(counts)
    if n == 0 or r == 0 or r == n or len(counts) < 3:
        return {"z": 0.0, "p": 1.0, "slope_pp": 0.0}
    x = list(range(len(counts)))
    xbar = sum(t * xi for t, xi in zip(totals, x)) / n
    pbar = r / n
    t_stat = sum((ci - ti * pbar) * (xi - xbar) for ci, ti, xi in zip(counts, totals, x))
    var = pbar * (1 - pbar) * sum(ti * (xi - xbar) ** 2 for ti, xi in zip(totals, x))
    if var <= 0:
        return {"z": 0.0, "p": 1.0, "slope_pp": 0.0}
    z = t_stat / math.sqrt(var)
    # Least-squares slope on the observed shares, for a readable effect size.
    shares = [c / t if t else 0.0 for c, t in zip(counts, totals)]
    m = len(shares)
    mx = (m - 1) / 2
    my = sum(shares) / m
    den = sum((i - mx) ** 2 for i in range(m))
    slope = sum((i - mx) * (s - my) for i, s in enumerate(shares)) / den if den else 0.0
    return {"z": round(z, 3), "p": round(_norm_sf(z), 5), "slope_pp": round(slope * 100, 2)}


@dataclass
class SkillRow:
    skill: str
    months: list[str]
    counts: list[int]
    totals: list[int]
    shares: list[float]
    lo: list[float]
    hi: list[float]
    first_share: float
    last_share: float
    change_pp: float
    trend: dict
    total_mentions: int

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "shares": [round(x, 4) for x in self.shares],
            "lo": [round(x, 4) for x in self.lo],
            "hi": [round(x, 4) for x in self.hi],
            "counts": self.counts,
            "first_share": round(self.first_share, 4),
            "last_share": round(self.last_share, 4),
            "change_pp": round(self.change_pp, 2),
            "trend": self.trend,
            "total_mentions": self.total_mentions,
            "significant": self.trend["p"] < 0.05,
            "robust": bool(self.trend.get("fdr_robust", False)),
        }


def month_of(panel: Panel, idx: int) -> str:
    return panel.run_dates[idx].strftime("%Y-%m")


def skill_index(panel: Panel, spells: list[Spell], basis: str = "adverts") -> dict:
    """Monthly demand for each skill, with trend tests.

    Two bases, because they answer different questions and only together are
    they honest.

    ``adverts`` is the share of described vacancies naming a skill. It is the
    intuitive reading, but it is dragged down for every skill by adverts
    becoming less specific over time: mean named tools per advert fell from
    2.31 to 1.75 across this window.

    ``mentions`` is the skill's share of all skill mentions that month. It
    removes that drift and isolates composition: given that an advert names
    tools at all, which tools does it name. A skill can fall on the first
    basis and rise on the second, and that combination is a real and
    interpretable result rather than a contradiction.
    """
    uni = described(spells)
    totals_by_month: dict[str, int] = defaultdict(int)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for s in uni:
        m = month_of(panel, s.start_idx)
        sk = _skills_of(s)
        if basis == "mentions":
            if not sk:
                continue
            totals_by_month[m] += len(sk)
        else:
            totals_by_month[m] += 1
        for k in sk:
            counts[k][m] += 1

    floor = MIN_MENTION_SAMPLE if basis == "mentions" else MIN_MONTH_SAMPLE
    months = sorted(m for m, n in totals_by_month.items() if n >= floor)
    excluded = sorted(m for m, n in totals_by_month.items() if n < floor)
    totals = [totals_by_month[m] for m in months]

    rows: list[SkillRow] = []
    for sk, per_month in counts.items():
        c = [per_month.get(m, 0) for m in months]
        if sum(c) < MIN_SKILL_TOTAL:
            continue
        shares = [ci / ti for ci, ti in zip(c, totals)]
        bounds = [wilson(ci, ti) for ci, ti in zip(c, totals)]
        rows.append(
            SkillRow(
                skill=sk,
                months=months,
                counts=c,
                totals=totals,
                shares=shares,
                lo=[b[0] for b in bounds],
                hi=[b[1] for b in bounds],
                first_share=shares[0],
                last_share=shares[-1],
                change_pp=(shares[-1] - shares[0]) * 100,
                trend=cochran_armitage(c, totals),
                total_mentions=sum(c),
            )
        )

    rows.sort(key=lambda r: -r.last_share)

    # Twenty-odd technologies each get a trend test, so at p<0.05 about one
    # false positive is expected by chance alone. Benjamini-Hochberg controls
    # the false discovery rate at 5% across the family; a technology is only
    # called a robust mover if it clears that, and the nominal result is kept
    # separately so the reader can see which claims would not survive it.
    m = len(rows)
    ranked = sorted(rows, key=lambda r: r.trend["p"])
    cutoff = 0
    for i, r in enumerate(ranked, 1):
        if r.trend["p"] <= 0.05 * i / m:
            cutoff = i
    robust = {id(r) for r in ranked[:cutoff]}
    for r in rows:
        r.trend["fdr_robust"] = id(r) in robust
        r.trend["bonferroni_robust"] = r.trend["p"] < 0.05 / m

    return {
        "basis": basis,
        "n_tests": m,
        "fdr_method": "Benjamini-Hochberg, FDR 5%",
        "months": months,
        "month_totals": totals,
        "excluded_months": excluded,
        "universe": (
            "skill mentions in described vacancies"
            if basis == "mentions"
            else "vacancies whose advert text was retrieved"
        ),
        "n_universe": len(uni),
        "n_all": len(spells),
        "skills": [r.to_dict() for r in rows],
    }


def rank_movement(index: dict) -> list[dict]:
    """Rank of each skill in the first and last comparable month.

    Ranks rather than shares, because the reader's question is which skills
    employers name most often relative to each other, and ranks are robust to
    the residual differences in text retrieval between months.
    """
    skills = index["skills"]
    if not skills:
        return []
    first = sorted(skills, key=lambda s: -s["shares"][0])
    last = sorted(skills, key=lambda s: -s["shares"][-1])
    r_first = {s["skill"]: i + 1 for i, s in enumerate(first)}
    r_last = {s["skill"]: i + 1 for i, s in enumerate(last)}
    return sorted(
        [
            {
                "skill": s["skill"],
                "rank_first": r_first[s["skill"]],
                "rank_last": r_last[s["skill"]],
                "move": r_first[s["skill"]] - r_last[s["skill"]],
                "share_first": s["shares"][0],
                "share_last": s["shares"][-1],
                "significant": s["significant"],
            }
            for s in skills
        ],
        key=lambda d: d["rank_last"],
    )


def cooccurrence(spells: list[Spell], top: list[str], min_pair: int = 8) -> dict:
    """How often skills are named together, as lift over independence.

    Raw co-occurrence counts just re-rank the most common skills against each
    other. Lift answers the more useful question: given an advert mentions one,
    how much more likely is the other than chance.
    """
    uni = described(spells)
    n = len(uni)
    sets = [_skills_of(s) for s in uni]
    single = {t: sum(1 for st in sets if t in st) for t in top}
    out = []
    for i, a in enumerate(top):
        for b in top[i + 1 :]:
            both = sum(1 for st in sets if a in st and b in st)
            if both < min_pair:
                continue
            expected = single[a] * single[b] / n if n else 0
            if expected <= 0:
                continue
            out.append(
                {
                    "a": a,
                    "b": b,
                    "both": both,
                    "lift": round(both / expected, 2),
                    "given_a": round(both / single[a], 3) if single[a] else 0,
                }
            )
    out.sort(key=lambda d: -d["lift"])
    return {"pairs": out[:26], "n_universe": n, "singles": single}


def composition(panel: Panel, spells: list[Spell], field: str) -> dict:
    """Monthly mix of a categorical advert attribute.

    Uses all vacancies, not only described ones: seniority, remote status and
    sector are parsed from the listing card rather than the advert body, so
    they do not carry the text-retrieval bias that skills do.
    """
    totals: dict[str, int] = defaultdict(int)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in spells:
        v = (getattr(s, field, "") or "").strip().lower()
        if not v or v == "unknown":
            continue
        m = month_of(panel, s.start_idx)
        totals[m] += 1
        counts[v][m] += 1

    months = sorted(m for m, n in totals.items() if n >= MIN_MONTH_SAMPLE)
    if not months:
        return {"months": [], "series": {}}
    tot = [totals[m] for m in months]
    series = {
        v: [round(per.get(m, 0) / t, 4) for m, t in zip(months, tot)]
        for v, per in counts.items()
        if sum(per.values()) >= 25
    }
    order = sorted(series, key=lambda v: -sum(series[v]))
    return {
        "months": months,
        "month_totals": tot,
        "series": {v: series[v] for v in order},
        "field": field,
    }


def top_employers(spells: list[Spell], n: int = 14) -> list[dict]:
    """Who is advertising most, and how long their adverts run."""
    by: dict[str, list[Spell]] = defaultdict(list)
    for s in spells:
        c = (s.company or "").strip()
        if c:
            by[c].append(s)
    rows = []
    for c, ss in by.items():
        if len(ss) < 4:
            continue
        days = sorted(x.duration_days for x in ss)
        rows.append(
            {
                "company": c,
                "vacancies": len(ss),
                "median_days": days[len(days) // 2],
                "reposts": round(sum(x.urls for x in ss) / len(ss), 1),
            }
        )
    rows.sort(key=lambda d: -d["vacancies"])
    return rows[:n]


def specificity(panel: Panel, spells: list[Spell]) -> dict:
    """How many tools an advert names, by month.

    This is the drift that forces the two-basis treatment above, so it is
    reported directly rather than left as a footnote.
    """
    uni = described(spells)
    per: dict[str, list[int]] = defaultdict(list)
    for s in uni:
        per[month_of(panel, s.start_idx)].append(len(_skills_of(s)))
    months = sorted(m for m, v in per.items() if len(v) >= MIN_MONTH_SAMPLE)
    return {
        "months": months,
        "mean_skills": [round(sum(per[m]) / len(per[m]), 3) for m in months],
        "share_naming_none": [
            round(sum(1 for x in per[m] if x == 0) / len(per[m]), 4) for m in months
        ],
        "n": [len(per[m]) for m in months],
    }
