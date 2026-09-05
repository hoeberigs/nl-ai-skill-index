# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Static HTML/CSS/JS, no framework and no chart library. Figures are hand-authored
inline SVG built from a generated `docs/data.json`. Published on GitHub Pages
from `/docs`, rebuilt weekly by a GitHub Action. Existing codebase; not a
greenfield decision.

## Users

Anyone with a stake in the Dutch AI job market: candidates deciding which
skills to invest in, recruiters and hiring managers calibrating what to ask
for, and journalists or analysts who want a defensible number. They arrive
cold from a link, usually from LinkedIn, with no prior context about the data
source and no obligation to stay.

## Product Purpose

Answer one question honestly: what are Dutch employers actually asking for in
AI, ML and data roles, and what has changed. Success is a reader who leaves
with a specific, correctly-caveated finding they can repeat and cite, rather
than a general impression.

## Positioning

Built on six months of *daily* snapshots of the same job boards, not a single
scrape. That panel structure is the mechanism a neighbouring product cannot
copy from a one-off crawl: it makes advert lifetime observable, which makes
"hardest to fill" measurable, and it exposes the scraper's own visibility bias
so the numbers can be corrected rather than quietly wrong.

## Constraints and durable facts

- Every published figure is measured from the panel and reproducible from the
  committed pipeline. No estimates presented as measurements.
- Statistical honesty is load-bearing, not decoration: trend tests, p-values,
  confidence intervals, the chance of misreading, and the limits section stay
  visible. The strongest finding in the piece is a *negative* one (most
  technologies did not move), and the design must be able to say that.
- Skill figures use adverts whose text was retrieved as the denominator, never
  the full panel, because retrieval coverage rose over the period.
- An advert disappearing is not a hire. The piece measures time advertised,
  never time to hire, and must never imply otherwise.
- Data refreshes weekly and every headline number is rendered from
  `data.json` at runtime. No finding may be hard-coded into the markup.

## Terminology

- **Vacancy** — a normalised (title, employer) identity, merging recruiter
  repost chains.
- **Spell** — one continuous advertising episode of a vacancy.
- **Share of mentions** — a technology's share of all technology mentions that
  month; the compositional basis that removes the drift in how specific
  adverts are.
- **RMST-90** — restricted mean survival time: average days advertised within
  90 days.

## Accessibility

Palettes are validated for CVD separation and contrast in light and dark before
shipping. Colour never carries meaning alone; every encoded state also has a
label, a symbol or a table row.

## Open decisions

- Visitor mode and visual world for this surface are recorded in the surface
  brief and DESIGN.md, not here.
