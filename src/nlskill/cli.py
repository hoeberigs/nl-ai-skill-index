"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from . import config as cfg
from .pipeline import run


def _resolve_db(arg: str | None) -> Path:
    """Find the tracker database, cloning the upstream repo if needed."""
    if arg:
        p = Path(arg).expanduser()
        if not p.exists():
            sys.exit(f"database not found: {p}")
        return p

    for candidate in (
        Path("data/jobs.db"),
        Path("../ai-job-tracker-nl/data/jobs.db"),
    ):
        if candidate.exists():
            return candidate

    dest = Path(tempfile.mkdtemp(prefix="tracker-")) / "ai-job-tracker-nl"
    print(f"cloning {cfg.TRACKER_REPO} ...", file=sys.stderr)
    subprocess.run(
        ["git", "clone", "--depth", "1", cfg.TRACKER_REPO, str(dest)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return dest / cfg.TRACKER_DB_RELPATH


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nl-ai-skill-index", description=__doc__)
    ap.add_argument("--db", help="path to the tracker's jobs.db")
    ap.add_argument("--out", default=str(cfg.DATA_JSON), help="dashboard JSON output")
    ap.add_argument("--cache", default=".cache", help="embedding cache directory")
    ap.add_argument("--quick", action="store_true", help="single calibration seed")
    args = ap.parse_args(argv)

    db = _resolve_db(args.db)
    print(f"database : {db}", file=sys.stderr)
    payload = run(db, Path(args.out), Path(args.cache), quick=args.quick)

    cal = payload["calibration"]
    det = payload["detection"]
    print(json.dumps(
        {
            "runs": payload["panel"]["n_runs"],
            "vacancies": payload["panel"]["n_identities"],
            "detection": det["marginal"],
            "persistence_odds_ratio": det["markov"]["odds_ratio"],
            "observed_median_days": cal["observed"]["median_span"],
            "true_median_days": cal["median_true_days"],
            "ephemeral_share": cal["ephemeral_share"],
            "clusters": payload["clusters"]["k"],
            "classifier_macro_f1": payload["classifier"]["macro_f1"],
            "rescued": payload["classifier"]["n_rescued"],
            "seconds": payload["runtime_seconds"],
        },
        indent=2,
    ), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
