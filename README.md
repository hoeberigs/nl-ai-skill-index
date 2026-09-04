# The NL AI Skill Index

**[Read the index →](https://hoeberigs.github.io/nl-ai-skill-index/)**

What Dutch employers actually ask for in AI, ML and data roles, measured from six months of
daily job-board snapshots rather than a single scrape.

The headline finding is not a technology winning. It is adverts getting **vaguer**: the average
Dutch AI advert named 2.41 specific technologies in April and 1.75 by August, a 27% drop, while
the share naming none at all rose from 30% to 39%. Against that backdrop only **RAG** measurably
gained share of what employers name, and **NLP** and **Kubernetes** measurably lost it. Everything
else moved no more than sampling noise.

## Why daily snapshots change the answer

A single scrape tells you what is advertised today. Capturing the same boards every day for six
months lets you follow each advert from the day it appears to the day it goes, which is what makes
"hardest to fill" measurable at all.

It also exposes a problem that a single scrape hides, and getting it wrong changes the numbers by
a factor of seven.

### The scraper sees about half of what is live

Between any two sightings of the same advert it was necessarily live throughout. Every day in
between is therefore a trial the scraper either passed or failed, and detection becomes measurable
with no external ground truth:

| Advert age | Detection rate |
|---|---|
| 1–6 days | 65% |
| 7–13 days | 60% |
| 14–29 days | 53% |
| 30–59 days | 43% |

Detection falls with exactly the variable being measured, so every naive duration statistic is
biased downwards.

### Misses come in runs, not at random

This is the part that is easy to get wrong. Detection is strongly autocorrelated:

- P(seen tomorrow | seen today) = **0.80**
- P(seen tomorrow | missed today) = **0.22**
- odds ratio **14.4**

An advert sits visible in the board's top results for a run of days, then buried for a run of days.
That matters because the rule for deciding an advert is gone is "absent for K days running", and the
false-delisting rate is not `(1-p)^K` but `(1 - P(seen | missed))^K`:

| K | Assuming independence | Measured |
|---|---|---|
| 7 | 0.4% | **18%** |
| 14 | 0.002% | **3.3%** |
| 21 | ~0% | **0.6%** |

The independence assumption makes K=14 look risk-free when it is roughly 15× riskier than stated.
This project uses **K=21**.

### Recovering true lifetimes

A grace period repairs gaps inside a sighting history but cannot repair the edges: a third of
adverts are seen exactly once, so their measured span is one day however long they really ran.

Rather than patching the estimate afterwards, the observation process is run forwards. A candidate
lifetime distribution is pushed through the measured two-state detection model and the same spell
splitting the real pipeline applies, and the parameters that reproduce the observed data are the
estimate (simulated method of moments). The fit is repeated across five seeds and the spread
reported.

A single log-normal could not reproduce a 33% single-sighting share alongside a 38-day upper decile.
A two-component mixture can, and it corresponds to something real: listings that appear once and
never establish themselves, and genuine vacancies that run for weeks.

**Result:** genuine vacancies run a median of ~14 days, against the 2 days a naive last-seen reading
of the same data reports.

## What the index measures

Skills are read from advert text, and text retrieval improved sharply over the period (20% coverage
in March, ~68% by summer). Two consequences drive the whole design:

1. **The denominator is always adverts whose text was retrieved**, never the full panel. Otherwise
   almost every skill appears to "rise" in April, which is the scraper improving.
2. **Two bases are reported.** Share of adverts is intuitive but is dragged down for every skill by
   adverts becoming less specific. Share of mentions removes that drift and isolates composition. A
   skill can fall on the first and rise on the second, and that combination is the actual result.

Trends use a **Cochran-Armitage test across all five months**, not a comparison of two endpoints, so
a result cannot be manufactured by choosing convenient months. Each point carries a Wilson interval.

## Roles

Role groupings are discovered from the adverts rather than taken from a fixed list: every vacancy is
embedded with a multilingual sentence encoder (Dutch and English appear side by side, often in the
same advert), reduced by PCA and clustered with k-means, k chosen by silhouette.

Density clustering was tried first and rejected: HDBSCAN left 38–45% of vacancies as noise at every
setting swept, which is a defensible statistical answer and a useless market map.

Silhouette scores sit near 0.15. That is a property of the data, not a tuning failure: adverts
occupy a continuum between adjacent roles. The clusters are useful summaries of a continuous space,
not evidence of discrete role types, and the score is published so readers can judge that.

## Data quality rules

Both are applied explicitly and their attrition is reported rather than absorbed silently:

- **Non-AI staffing noise** (16.7% of the panel) is removed from the demand index. A keyword-driven
  AI job scraper also catches production and operator adverts sitting near the query terms; the
  largest employers by advert count were temp agencies. They are kept in the survival analysis,
  where the contrast is informative: agency adverts have a median life of 1 day against roughly two
  weeks for genuine vacancies.
- **Template adverts** containing lorem ipsum filler are dropped.

## Install and run

```bash
git clone https://github.com/hoeberigs/nl-ai-skill-index.git
cd nl-ai-skill-index
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

```bash
python -m nlskill
```

With no `--db`, the upstream tracker repository is cloned automatically. Embeddings are cached, so
subsequent runs take seconds. `--quick` uses a single calibration seed.

```bash
python -m nlskill --db ../ai-job-tracker-nl/data/jobs.db --out docs/data.json
```

Then open `docs/index.html` over HTTP (the page fetches `data.json`, which `file://` blocks).

## Layout

```
src/nlskill/
  data.py         panel construction, vacancy identity, spell splitting
  detection.py    detection curve and the two-state visibility model
  survival.py     Kaplan-Meier, log-log bands, log-rank
  calibrate.py    simulated method of moments for true lifetimes
  market.py       the demand index, trend tests, co-occurrence, employers
  cluster.py      embeddings to role archetypes, c-TF-IDF labelling
  classify.py     recovering the unlabelled "other" bucket
  trends.py       weekly inflow and stock
  pipeline.py     runs everything, emits docs/data.json
docs/
  index.html      the published index (hand-built SVG, no chart library)
  data.json       generated
```

## Limits

- **An advert disappearing is not a hire.** Roles leave a board when filled, withdrawn, expired or
  defunded, and this data cannot separate those. Everything here measures time advertised, never
  time to hire.
- **Two boards are not the market.** Roles filled through networks, agencies or internal moves never
  appear.
- **Vacancy identity merges same-title roles at one employer.** A company genuinely hiring three data
  scientists appears as one long-running vacancy.
- **Skill vocabulary is inherited** from the upstream tracker's keyword tagger, so it captures named
  technologies rather than everything an advert asks for.

## Data source

Built on [ai-job-tracker-nl](https://github.com/hoeberigs/ai-job-tracker-nl), which scrapes Dutch job
boards daily and commits its SQLite database, making it a versioned and reproducible input.

## Licence

MIT. See [LICENSE](LICENSE).
