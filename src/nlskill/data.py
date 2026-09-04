"""Turn the tracker's snapshot table into a vacancy panel.

The upstream database stores one row per (posting, scrape run). Counting those
rows measures scraping effort, not the job market, so everything here works
from a panel keyed on the vacancy with an explicit list of the days it was
seen.
"""

from __future__ import annotations

import datetime as dt
import re
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Recruiters decorate the same role with locations, contract types and
# marketing suffixes. Stripping them is what allows a repost chain to be
# recognised as one vacancy.
_NOISE = re.compile(
    r"\b(m/v/x|m/f/d|m/v|fulltime|full[- ]time|parttime|part[- ]time|"
    r"freelance|zzp|interim|vacature|vacancy|hybrid|remote|onsite|"
    r"per direct|urgent|new|nieuw)\b",
    re.I,
)
_BRACKETS = re.compile(r"[\(\[\{].*?[\)\]\}]")
_NONWORD = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")

# Placeholder adverts published from an unfinished template. Few in number,
# but they carry no role information and their Latin filler competes for
# cluster keywords, so they are removed and counted rather than left in.
_PLACEHOLDER = re.compile(r"lorem ipsum|dolor sit amet|consectetur adipiscing", re.I)


def is_placeholder(description: str) -> bool:
    return bool(description) and bool(_PLACEHOLDER.search(description))


def normalise_title(title: str) -> str:
    """Reduce a job title to a comparable core."""
    if not title:
        return ""
    s = unicodedata.normalize("NFKD", title)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = _BRACKETS.sub(" ", s)
    s = s.replace("&", " and ")
    s = _NOISE.sub(" ", s)
    s = _NONWORD.sub(" ", s)
    return _WS.sub(" ", s).strip()


def normalise_company(company: str) -> str:
    if not company:
        return ""
    s = unicodedata.normalize("NFKD", company)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\b(b\.?v\.?|n\.?v\.?|inc|ltd|gmbh|group|nederland)\b", " ", s)
    s = _NONWORD.sub(" ", s)
    return _WS.sub(" ", s).strip()


@dataclass
class Spell:
    """One continuous advertising episode for one vacancy."""

    key: str
    title: str
    company: str
    start_idx: int
    last_seen_idx: int
    sightings: int
    left_censored: bool
    right_censored: bool
    category: str = ""
    seniority: str = ""
    sector: str = ""
    remote: str = ""
    skills: str = ""
    description: str = ""
    urls: int = 1

    @property
    def duration_days(self) -> int:
        """Days from first to last sighting, inclusive."""
        return self.last_seen_idx - self.start_idx + 1

    @property
    def observed_event(self) -> bool:
        """True when the vacancy was seen to disappear rather than censored."""
        return not self.right_censored


@dataclass
class Panel:
    run_dates: list[dt.date]
    spells: list[Spell] = field(default_factory=list)
    # vacancy key -> sorted run indices where it was sighted
    histories: dict[str, list[int]] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        return len(self.run_dates)


def _row_meta(rows: list[dict]) -> dict:
    """Pick the most informative variant across a repost chain."""
    best = max(rows, key=lambda r: len(r.get("description") or ""))
    out = {
        k: (best.get(k) or "")
        for k in ("category", "seniority", "sector", "remote", "skills", "description")
    }
    # A chain is only "other" if every variant was unclassifiable.
    cats = [r.get("category") for r in rows if r.get("category") not in (None, "", "other")]
    if cats:
        out["category"] = max(set(cats), key=cats.count)
    return out


def load_panel(db_path: Path, grace_days: int, key: str = "vacancy") -> Panel:
    """Build a vacancy panel from the tracker database.

    ``key='vacancy'`` merges recruiter repost chains under one identity.
    ``key='url'`` keeps every posting URL separate, which is what a naive
    reading of the data would do, and is retained for comparison.
    """
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    run_date = {
        r["id"]: dt.date.fromisoformat(r["run_date"])
        for r in con.execute("SELECT id, run_date FROM scrape_runs")
    }
    run_dates = sorted(set(run_date.values()))
    idx_of = {d: i for i, d in enumerate(run_dates)}
    n_runs = len(run_dates)

    sightings: dict[str, set[int]] = defaultdict(set)
    rows_by_key: dict[str, list[dict]] = defaultdict(list)
    urls_by_key: dict[str, set[str]] = defaultdict(set)
    label: dict[str, tuple[str, str]] = {}

    sql = """
        SELECT run_id, title, company, url, category, seniority,
               sector, remote, skills, description
        FROM jobs
    """
    for r in con.execute(sql):
        rid = r["run_id"]
        if rid not in run_date:
            continue
        title = (r["title"] or "").strip()
        company = (r["company"] or "").strip()
        if key == "url":
            k = r["url"] or f"{title}|{company}"
        else:
            nt, nc = normalise_title(title), normalise_company(company)
            if not nt:
                continue
            k = f"{nt}@@{nc}"
        sightings[k].add(idx_of[run_date[rid]])
        urls_by_key[k].add(r["url"] or "")
        label.setdefault(k, (title, company))
        rows_by_key[k].append(dict(r))
    con.close()

    panel = Panel(run_dates=run_dates)
    panel.histories = {k: sorted(v) for k, v in sightings.items()}

    for k, seen in panel.histories.items():
        title, company = label[k]
        meta = _row_meta(rows_by_key[k])
        # Split the sighting history wherever the vacancy was absent for at
        # least `grace_days` consecutive runs. Shorter absences are treated as
        # the scraper missing a live posting, which the detection analysis
        # shows happens about a third of the time on any given day.
        blocks: list[list[int]] = [[seen[0]]]
        for prev, cur in zip(seen, seen[1:]):
            if cur - prev - 1 >= grace_days:
                blocks.append([cur])
            else:
                blocks[-1].append(cur)

        for block in blocks:
            start, last = block[0], block[-1]
            # Still live at the end of observation if the trailing silence is
            # too short to conclude the vacancy was withdrawn.
            right_cens = (n_runs - 1 - last) < grace_days
            panel.spells.append(
                Spell(
                    key=k,
                    title=title,
                    company=company,
                    start_idx=start,
                    last_seen_idx=last,
                    sightings=len(block),
                    left_censored=(start == 0),
                    right_censored=right_cens,
                    urls=len(urls_by_key[k]),
                    **meta,
                )
            )

    n_placeholder = sum(1 for sp in panel.spells if is_placeholder(sp.description))
    if n_placeholder:
        panel.spells = [sp for sp in panel.spells if not is_placeholder(sp.description)]

    panel.meta = {
        "placeholder_spells_dropped": n_placeholder,
        "key": key,
        "grace_days": grace_days,
        "n_runs": n_runs,
        "first_run": run_dates[0].isoformat(),
        "last_run": run_dates[-1].isoformat(),
        "n_identities": len(panel.histories),
        "n_spells": len(panel.spells),
    }
    return panel
