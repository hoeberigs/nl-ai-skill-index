"""Encode vacancies into a multilingual semantic space.

Dutch and English listings sit side by side in this market, often inside the
same advert, so a monolingual encoder would split identical roles by language
before any analysis begins. Embeddings are cached to disk because they are
deterministic given the model and the input text, and recomputing them
dominates the runtime of every downstream step.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .config import EMBED_BATCH, EMBED_MODEL, MAX_CHARS
from .data import Spell


def vacancy_text(spell: Spell) -> str:
    """Compose the text used to represent a vacancy.

    The title is repeated deliberately. Roughly half these vacancies carry no
    description at all, and without the repetition the ones that do would be
    dominated by boilerplate about secondary benefits and equal-opportunity
    statements, pulling unrelated roles together on shared legal language.
    """
    title = (spell.title or "").strip()
    desc = (spell.description or "").strip()
    parts = [title, title]
    if spell.company:
        parts.append(spell.company)
    if desc:
        parts.append(desc[:MAX_CHARS])
    return " . ".join(p for p in parts if p)


def _fingerprint(texts: list[str]) -> str:
    h = hashlib.sha256()
    h.update(EMBED_MODEL.encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def embed_texts(texts: list[str], cache_dir: Path, show_progress: bool = True) -> np.ndarray:
    """Return L2-normalised embeddings, reusing a cached matrix when valid."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"emb_{_fingerprint(texts)}.npy"
    if cache.exists():
        arr = np.load(cache)
        if arr.shape[0] == len(texts):
            return arr

    # Imported lazily so that steps not needing embeddings stay fast to start.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    arr = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(cache, arr)
    return arr
