"""Run the full analysis and emit the dashboard payload."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import config as cfg
from .calibrate import calibrate_robust, observed_targets
from .classify import rescue_other
from .cluster import cluster_vacancies
from .data import Panel, Spell, load_panel
from .detection import estimate_detection, estimate_markov, grace_period_table
from .market import (
    composition,
    cooccurrence,
    filter_ai,
    provider_shares,
    rank_movement,
    skill_index,
    specificity,
    top_employers,
)
from .embed import embed_texts, vacancy_text
from .survival import kaplan_meier, logrank
from .trends import skill_trends, trend_slope, weekly_series


def representative_spells(panel: Panel) -> list[Spell]:
    """One row per vacancy identity, taking its longest advertising episode."""
    best: dict[str, Spell] = {}
    for s in panel.spells:
        cur = best.get(s.key)
        if cur is None or s.duration_days > cur.duration_days:
            best[s.key] = s
    return list(best.values())


def run(db_path: Path, out_path: Path, cache_dir: Path, quick: bool = False) -> dict:
    t0 = time.time()
    payload: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # --- Panel and instrument ---------------------------------------------
    panel = load_panel(db_path, cfg.GRACE_DAYS, key="vacancy")
    analysis = [s for s in panel.spells if not s.left_censored]
    dropped = len(panel.spells) - len(analysis)

    curve = estimate_detection(panel)
    markov = estimate_markov(panel)

    payload["panel"] = {
        **panel.meta,
        "n_spells_analysed": len(analysis),
        "n_left_censored_dropped": dropped,
        "n_right_censored": sum(1 for s in analysis if s.right_censored),
    }
    payload["detection"] = {
        "marginal": round(curve.overall(), 4),
        "worst_case": round(curve.worst_case(), 4),
        "by_age": curve.binned(),
        "markov": {
            "p_stay_visible": round(markov.p_stay_visible, 4),
            "p_become_visible": round(markov.p_become_visible, 4),
            "stationary": round(markov.stationary, 4),
            "odds_ratio": round(markov.odds_ratio, 2),
            "n_transitions": markov.n_transitions,
        },
        "grace_table": grace_period_table(curve, markov),
        "grace_chosen": cfg.GRACE_DAYS,
    }

    # --- Survival, and the sensitivity that shows it is not threshold-driven
    km_all = kaplan_meier(analysis)
    payload["survival_overall"] = km_all.as_dict()
    payload["survival_overall"]["rmst_90"] = round(km_all.restricted_mean(90), 2)

    sens = []
    for k in cfg.GRACE_SENSITIVITY:
        p_k = load_panel(db_path, k, key="vacancy")
        a_k = [s for s in p_k.spells if not s.left_censored]
        km_k = kaplan_meier(a_k)
        sens.append(
            {
                "grace_days": k,
                "n": km_k.n,
                "median": km_k.median,
                "rmst_90": round(km_k.restricted_mean(90), 2),
                "false_delisting_markov": round(markov.false_delisting_rate(k), 4),
            }
        )
    payload["grace_sensitivity"] = sens

    # --- Recovering true lifetimes ----------------------------------------
    obs = observed_targets(analysis)
    seeds = (42,) if quick else (1, 7, 42, 99, 2026)
    payload["calibration"] = calibrate_robust(
        obs, curve, markov, panel.n_runs, cfg.GRACE_DAYS,
        seeds=seeds, n_sim=4000 if quick else 6000,
    )
    payload["calibration"]["observed"] = {k: round(v, 4) for k, v in obs.items()}

    # --- Semantics: embed, cluster, rescue --------------------------------
    reps = representative_spells(panel)
    texts = [vacancy_text(s) for s in reps]
    titles = [s.title for s in reps]
    emb = embed_texts(texts, cache_dir, show_progress=False)

    clusters = cluster_vacancies(emb, texts, titles)
    payload["clusters"] = {
        "k": clusters.n_clusters,
        "silhouette": clusters.silhouette,
        "silhouette_by_k": clusters.silhouette_by_k,
        "names": {str(c): n for c, n in clusters.names.items()},
        "keywords": {str(c): kw for c, kw in clusters.keywords.items()},
        "sizes": {str(c): n for c, n in clusters.sizes.items()},
    }

    rescue = rescue_other(emb, [s.category for s in reps])
    payload["classifier"] = {
        "macro_f1": rescue.macro_f1,
        "accuracy": rescue.accuracy,
        "n_train": rescue.n_train,
        "n_other": rescue.n_other,
        "n_rescued": rescue.n_rescued,
        "rescue_rate": rescue.rescue_rate,
        "threshold": cfg.RESCUE_THRESHOLD,
        "rescued_breakdown": rescue.rescued_breakdown,
        "per_class": {
            k: v for k, v in rescue.per_class.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
    }

    # --- Survival by cluster, the headline comparison ----------------------
    cluster_of = {s.key: int(c) for s, c in zip(reps, clusters.labels)}
    by_cluster: dict[int, list[Spell]] = {}
    for s in analysis:
        c = cluster_of.get(s.key)
        if c is not None:
            by_cluster.setdefault(c, []).append(s)

    cluster_curves = []
    for c, spells in sorted(by_cluster.items(), key=lambda kv: -len(kv[1])):
        if len(spells) < 40:
            continue
        km = kaplan_meier(spells)
        others = [s for cc, ss in by_cluster.items() if cc != c for s in ss]
        cluster_curves.append(
            {
                "cluster": c,
                "name": clusters.names.get(c, str(c)),
                "n": km.n,
                "median": km.median,
                "rmst_90": round(km.restricted_mean(90), 2),
                "times": km.times,
                "survival": [round(x, 5) for x in km.survival],
                "logrank_vs_rest": logrank(spells, others),
            }
        )
    cluster_curves.sort(key=lambda d: -d["rmst_90"])
    payload["survival_by_cluster"] = cluster_curves

    # --- The demand index -------------------------------------------------
    # Non-AI staffing noise is removed here and only here: the survival and
    # detection sections above deliberately keep it, because the contrast
    # between agency churn and genuine vacancies is itself informative.
    ai_spells, attrition = filter_ai(analysis)
    payload["relevance_filter"] = attrition

    idx_adverts = skill_index(panel, ai_spells, basis="adverts")
    idx_mentions = skill_index(panel, ai_spells, basis="mentions")
    payload["skill_index"] = {
        "adverts": idx_adverts,
        "mentions": idx_mentions,
        "rank_movement": rank_movement(idx_mentions),
        "specificity": specificity(panel, ai_spells),
    }
    top_skills = [r["skill"] for r in idx_mentions["skills"][:14]]
    payload["cooccurrence"] = cooccurrence(ai_spells, top_skills)
    payload["composition"] = {
        f: composition(panel, ai_spells, f) for f in ("seniority", "remote", "sector")
    }
    payload["employers"] = top_employers(ai_spells)
    payload["providers"] = provider_shares(panel, ai_spells)

    # --- Composition over time --------------------------------------------
    group_of = {s.key: clusters.names.get(cluster_of.get(s.key, -1), "Other") for s in reps}
    payload["weekly"] = weekly_series(panel, analysis, group_of)
    payload["skills"] = skill_trends(panel, analysis)
    payload["skills"]["slopes"] = {
        k: round(trend_slope(v) * 100, 4) for k, v in payload["skills"]["series"].items()
    }

    # --- Embedding map ----------------------------------------------------
    coords = clusters.coords
    lim = np.percentile(np.abs(coords), 99) or 1.0
    step = max(1, len(reps) // 1400)
    payload["map"] = [
        {
            "x": round(float(coords[i, 0] / lim), 4),
            "y": round(float(coords[i, 1] / lim), 4),
            "c": int(clusters.labels[i]),
            "t": reps[i].title[:70],
            "co": reps[i].company[:40],
            "d": reps[i].duration_days,
        }
        for i in range(0, len(reps), step)
    ]

    # --- Longest-running vacancies, read rather than assumed ---------------
    longest = sorted(analysis, key=lambda s: -s.duration_days)[:25]
    payload["longest"] = [
        {
            "title": s.title[:80],
            "company": s.company[:45],
            "days": s.duration_days,
            "sightings": s.sightings,
            "cluster": clusters.names.get(cluster_of.get(s.key, -1), ""),
            "censored": s.right_censored,
            "urls": s.urls,
        }
        for s in longest
    ]

    payload["runtime_seconds"] = round(time.time() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    return payload
