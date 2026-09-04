"""Discover role archetypes from the vacancies themselves.

The upstream tracker assigns categories with keyword rules, which is fast and
transparent but leaves a fifth of postings in an "other" bucket and cannot
surface a role it was never told about. Clustering the embeddings instead lets
the structure come from the market: whatever roles Dutch employers are
actually advertising will separate, named or not.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

from .config import CLUSTER_DIMS, CLUSTER_K_RANGE, RANDOM_STATE

# Dutch and English stop words, plus the recruitment boilerplate that
# otherwise wins every keyword ranking.
_STOP = set(
    """
de het een en van in op voor met te dat die is zijn we wij je jij jouw ons onze
u uw als aan bij naar door over uit om ook maar of dan er heeft hebben wordt
worden kan kunnen zal zullen niet geen meer veel zeer goed nieuw werk werken
functie bedrijf team collega collega's ervaring kennis binnen waar deze dit
the a an and or of to in for with on at as is are be been we you your our us
their this that will can may have has had from by about into more most very
you'll we're role job position company team work working experience years
knowledge skills strong good great new opportunity looking join apply
candidate candidates ideal must should would could well including etc
""".split()
)


@dataclass
class ClusterResult:
    labels: np.ndarray
    names: dict[int, str]
    keywords: dict[int, list[str]]
    sizes: dict[int, int]
    coords: np.ndarray
    n_clusters: int
    silhouette: float
    silhouette_by_k: dict[int, float]


def _ctfidf_keywords(texts: list[str], labels: np.ndarray, top_k: int = 8) -> dict[int, list[str]]:
    """Class-based TF-IDF: what distinguishes a cluster from the whole corpus.

    Plain per-cluster term frequency returns the same generic vocabulary for
    every cluster. Weighting by how concentrated a term is in one cluster
    relative to the corpus is what makes the labels discriminative.
    """
    uniq = sorted(set(labels.tolist()) - {-1})
    if not uniq:
        return {}
    joined = {c: " ".join(t for t, l in zip(texts, labels) if l == c) for c in uniq}
    vec = CountVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=40000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#./-]{1,}\b",
    )
    # Densified deliberately: this matrix is one row per cluster, so it is
    # tiny, and the ranking below needs ordinary array semantics.
    X = np.asarray(vec.fit_transform([joined[c] for c in uniq]).todense(), dtype=np.float64)
    vocab = np.array(vec.get_feature_names_out())

    keep = np.array(
        [
            not any(tok in _STOP for tok in term.split())
            for term in vocab
        ]
    )
    X = X[:, keep]
    vocab = vocab[keep]

    tf = X / np.maximum(X.sum(axis=1, keepdims=True), 1e-9)
    df = (X > 0).sum(axis=0)
    idf = np.log(1.0 + len(uniq) / np.maximum(df, 1e-9))
    scores = tf * idf

    out: dict[int, list[str]] = {}
    for i, c in enumerate(uniq):
        order = np.argsort(scores[i])[::-1]
        picked: list[str] = []
        for j in order:
            term = vocab[j]
            # Skip a term already covered by a longer phrase already chosen.
            if any(term in p or p in term for p in picked):
                continue
            picked.append(term)
            if len(picked) >= top_k:
                break
        out[c] = picked
    return out


def _title_counts(titles: list[str]) -> list[tuple[str, int]]:
    norm = []
    for t in titles:
        if not t or not t.strip():
            continue
        x = re.sub(r"[\(\[].*?[\)\]]", " ", t.lower())
        x = re.sub(r"\b(m/v/x|m/f/d|m/v|fulltime|parttime|freelance|zzp|vacature)\b", " ", x)
        x = re.sub(r"[^a-z0-9+#/ -]+", " ", x)
        x = re.sub(r"\s+", " ", x).strip(" -/")
        if x:
            norm.append(x)
    return Counter(norm).most_common()


def _assign_names(
    labels: np.ndarray, titles: list[str], keywords: dict[int, list[str]]
) -> dict[int, str]:
    """Label each cluster with the most common job title it can still claim.

    Keyword-derived labels were tried first and abandoned. Class-based TF-IDF
    reliably surfaces "ai", "data" and "engineer" for every cluster, because
    those words describe the whole market, and once they are barred the next
    terms down are fragments rather than role names. Real job titles are both
    more readable and self-evidently faithful to the cluster's contents.

    Titles are claimed largest-cluster-first, so the biggest group keeps the
    plainest name and smaller neighbours fall through to the most frequent
    title that still distinguishes them.
    """
    clusters = sorted(set(labels.tolist()))
    counts = {c: _title_counts([t for t, l in zip(titles, labels) if l == c]) for c in clusters}

    names: dict[int, str] = {}
    claimed: set[str] = set()
    for c in sorted(clusters, key=lambda c: -int((labels == c).sum())):
        pick = ""
        for title, n in counts[c]:
            if title not in claimed and n >= 2:
                pick = title
                claimed.add(title)
                break
        if not pick:
            kws = [k for k in keywords.get(c, []) if k not in _STOP]
            pick = " / ".join(kws[:2]) if kws else f"cluster {c}"
        names[c] = pick.title()
    return names


def cluster_vacancies(
    embeddings: np.ndarray,
    texts: list[str],
    titles: list[str],
    n_components: int = CLUSTER_DIMS,
) -> ClusterResult:
    """Reduce, cluster, then label.

    K-means rather than a density method, chosen after testing both. HDBSCAN
    left 38-45% of vacancies as noise at every setting swept, which is a
    defensible statistical answer and a useless market map: the question here
    is what every advertised role looks like, not which ones sit in dense
    regions. K is selected by silhouette rather than fixed by hand.

    The silhouette scores are low in absolute terms, around 0.15. That is a
    real property of the data rather than a tuning failure: job adverts occupy
    a continuum between adjacent roles, and an "ML Engineer" advert shades into
    a "Data Scientist" one without a gap. The clusters are therefore useful
    summaries of a continuous space, not evidence of discrete role types, and
    the score is reported so a reader can judge that for themselves.
    """
    n_components = int(min(n_components, embeddings.shape[1], max(2, embeddings.shape[0] - 1)))
    reduced = PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(embeddings)

    scores: dict[int, float] = {}
    best_k, best_score, best_labels = None, -1.0, None
    for k in CLUSTER_K_RANGE:
        if k >= len(reduced):
            continue
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(reduced)
        score = float(
            silhouette_score(reduced, labels, sample_size=min(2000, len(reduced)), random_state=RANDOM_STATE)
        )
        scores[k] = round(score, 4)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    labels = best_labels if best_labels is not None else np.zeros(len(embeddings), dtype=int)

    keywords = _ctfidf_keywords(texts, labels)
    names = _assign_names(labels, titles, keywords)
    sizes = {c: int((labels == c).sum()) for c in sorted(set(labels.tolist()))}

    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(embeddings)

    return ClusterResult(
        labels=labels,
        names=names,
        keywords=keywords,
        sizes=sizes,
        coords=coords,
        n_clusters=len(sizes),
        silhouette=round(best_score, 4),
        silhouette_by_k=scores,
    )
