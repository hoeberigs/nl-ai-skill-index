"""Central configuration and tunable constants.

Every threshold here is a measurement decision, not a style choice, so each
carries the reasoning that set it.
"""

from pathlib import Path

# --- Input -----------------------------------------------------------------
# The upstream tracker repo commits its SQLite database, so it doubles as a
# reproducible, versioned input for this project.
TRACKER_REPO = "https://github.com/hoeberigs/ai-job-tracker-nl.git"
TRACKER_DB_RELPATH = Path("data") / "jobs.db"

# --- Survival model --------------------------------------------------------
# A posting is treated as delisted only after GRACE_DAYS consecutive daily
# runs without a sighting. The value is set from the measured detection
# probability, not by eye: the chance of missing a live posting K days in a
# row is (1 - p) ** K, evaluated at the worst-case p observed for the oldest
# postings (p ~ 0.30). That gives 8.2% false delistings at K=7 versus 0.68%
# at K=14, so K=14 is the smallest defensible threshold.
GRACE_DAYS = 21

# Reported alongside the headline to show the estimate is not an artefact of
# one threshold. K=1 is the naive "last seen" rule most published job-board
# tenure statistics implicitly use.
GRACE_SENSITIVITY = (1, 7, 14, 21)

# Spells already running on the first snapshot have unknown start dates.
# Including them would understate age, so they are dropped from the headline
# and reported separately.
DROP_LEFT_CENSORED = True

# --- Embeddings and clustering ---------------------------------------------
# Dutch and English appear in roughly equal measure in these listings, so the
# encoder has to be multilingual. This one is small enough to run on CPU in
# a GitHub Action.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH = 32
MAX_CHARS = 2000

# Role archetypes are discovered bottom-up rather than forced into the
# upstream keyword taxonomy. Dimensionality is reduced before clustering
# because distances in the raw 384-d space are too uniform to separate
# anything; 10 components retains the structure that matters here.
CLUSTER_DIMS = 10
CLUSTER_K_RANGE = tuple(range(6, 17))

# --- Supervised rescue -----------------------------------------------------
# Confidence a classifier must clear before overriding an "other" label.
RESCUE_THRESHOLD = 0.60
CV_FOLDS = 5
RANDOM_STATE = 42

# --- Output ----------------------------------------------------------------
DOCS_DIR = Path("docs")
DATA_JSON = DOCS_DIR / "data.json"
