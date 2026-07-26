"""In-memory ICD-10-CM fuzzy search over ``data/ICD10_DB.parquet``.

Why not PostgreSQL? The reference service (Experity/diagnosis-mcp) uses Postgres
with the ``pg_trgm`` extension for trigram fuzzy search. That requires a running
DB container and a load step. For a workshop tool the same job is done fully
in-memory with ``rapidfuzz`` over a pandas DataFrame — no Docker, no DDL.

``ICD10_DB.parquet`` is a pre-cleaned snapshot (see
``notebooks/diagnosis_analysis.py`` for how it was derived from the raw
``Diagnosis.parquet``). ``load_icd10_codes`` just reads and caches it.

Scorer note: the reference ranks on ``word_similarity(Long_Desc, query)``, which
is *asymmetric* — it finds the best-matching token subset of the query inside the
(longer) description. We match against ``Description`` with
``rapidfuzz.fuzz.token_set_ratio``, tolerant of the query being a token subset of
a longer description. On score ties we prefer the shorter (more general)
description, so e.g. "essential hypertension" ranks ``I10`` above its
pregnancy-complicating variants even though both tie at 100.

``token_set_ratio`` does saturate at 100 for every over-specific variant sharing
the query's tokens, in isolation — a prior default, ``fuzz.WRatio``, avoided that
specifically, but at a worse cost discovered via Langfuse traces in production:
its ``partial_token_set_ratio`` component saturates near 100 whenever a query and
an *unrelated* candidate merely share one short common token (e.g. "of"), so
qualifier-heavy queries (laterality/body site/encounter type) got confidently
wrong top matches with no relevant result anywhere nearby. ``token_set_ratio``
was benchmarked as better or equal on both a qualifier-heavy adversarial query
set and a general query set through the hybrid path (search_icd10_hybrid_batch)
that's actually exposed to agents — see notebooks/diagnosis_analysis.py's scorer
bake-off. Note it's a wash-to-slightly-worse than WRatio on the *pure fuzzy*
path (search_icd10/search_icd10_batch) in isolation, since that path has no FTS
ranking signal to absorb the subset-saturation ties; the hybrid RRF fusion's
independent FTS ranking is what keeps the tradeoff net-positive in production.
"""

import re
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process, utils

ICD10_PARQUET = Path("data/ICD10_DB.parquet")

# Default fuzzy-match scorer and preprocessing (lowercase, strip punctuation).
#
# fuzz.WRatio was the original default but has a specific failure mode on
# qualifier-heavy queries (laterality/body site/encounter type): its internal
# partial_token_set_ratio component saturates near 100 whenever the query and
# a candidate description share even one short common token (e.g. "of"),
# regardless of relevance, producing a flat, non-discriminating score for both
# nonsense matches and the true code alike. token_set_ratio doesn't have that
# component and was benchmarked as strictly better or equal on both a
# qualifier-heavy adversarial query set and the general query set (see
# notebooks/diagnosis_analysis.py's scorer bake-off).
_SCORER = fuzz.token_set_ratio
_PROCESSOR = utils.default_process

_FTS_TOKEN_RE = re.compile(r"\w+")


@lru_cache(maxsize=4)
def load_icd10_codes(path: str = str(ICD10_PARQUET)) -> pd.DataFrame:
    """Load the ICD-10-CM code snapshot.

    Cached so the parquet is read once per ``path``.
    """
    return pd.read_parquet(path)


def _row_to_item(row: pd.Series) -> dict:
    """Shape a DataFrame row into the search-result item (code + description)."""
    return {
        "icd10_code": row["ICD10_Code"],
        "description": row["Description"],
    }


def search_icd10(
    query: str,
    limit: int = 20,
    score_cutoff: float = 60.0,
    path: str = str(ICD10_PARQUET),
    scorer=_SCORER,
) -> list[dict]:
    """Fuzzy-search ICD-10-CM codes for a single diagnosis name.

    Args:
        query: Free-text diagnosis name, e.g. "community acquired pneumonia".
        limit: Max results to return.
        score_cutoff: Minimum similarity score (0-100) to include a match.
        path: Path to the ICD-10 code parquet.
        scorer: rapidfuzz scorer to use (see search_icd10_batch). Overridable
            for experimentation; production code should rely on the default.

    Returns:
        List of match dicts (icd10_code, description), each with an extra
        "score", ordered by descending similarity (shorter, more general
        descriptions win ties).
    """
    # Delegate to the batch path so single and batch rank identically (the
    # global length tiebreaker is applied before truncating to `limit`).
    return search_icd10_batch(
        [query],
        limit=limit,
        score_cutoff=score_cutoff,
        path=path,
        scorer=scorer,
    )[query]


def search_icd10_batch(
    queries: list[str],
    limit: int = 20,
    score_cutoff: float = 60.0,
    path: str = str(ICD10_PARQUET),
    scorer=_SCORER,
) -> dict[str, list[dict]]:
    """Fuzzy-search ICD-10-CM codes for many diagnosis names at once.

    Uses ``rapidfuzz.process.cdist`` to score all queries against all codes in
    one vectorized (multi-threaded) pass, then takes the top matches per query.

    Args:
        scorer: rapidfuzz scorer, e.g. ``fuzz.WRatio`` (the default) or
            ``fuzz.token_set_ratio``. Overridable so alternatives can be
            benchmarked (see notebooks/diagnosis_analysis.py) before changing
            the module-level default — WRatio's ``partial_token_set_ratio``
            component saturates near 100 whenever a query and a candidate
            description merely share one short common token (e.g. "of"),
            which produces irrelevant top matches for longer, qualifier-heavy
            queries (laterality/body site/encounter type).

    Returns:
        Mapping of each input query to its list of match dicts (same shape as
        ``search_icd10``).
    """
    df = load_icd10_codes(path)
    choices = df["Description"].tolist()
    desc_len = df["Description"].str.len().to_numpy()

    # Score matrix: shape (len(queries), len(choices)).
    scores = process.cdist(
        queries,
        choices,
        scorer=scorer,
        processor=_PROCESSOR,
        workers=-1,
    )

    results: dict[str, list[dict]] = {}
    for q_idx, query in enumerate(queries):
        row_scores = scores[q_idx]
        # Sort by score desc, then description length asc (prefer general codes).
        order = np.lexsort((desc_len, -row_scores))[:limit]
        items = []
        for idx in order:
            score = float(row_scores[idx])
            if score < score_cutoff:
                break
            item = _row_to_item(df.iloc[int(idx)])
            item["score"] = round(score, 1)
            items.append(item)
        results[query] = items
    return results


@lru_cache(maxsize=4)
def _build_fts5_index(path: str = str(ICD10_PARQUET)) -> sqlite3.Connection:
    """Build an in-memory SQLite FTS5 index over ICD-10 descriptions.

    In-memory analog of the Postgres ``tsvector``/``tsquery`` search in
    ``tools/icd10.py``: FTS5's default (bareword) query syntax requires all
    terms to match, like ``plainto_tsquery``, and its ``bm25()`` function
    provides the same kind of term-frequency relevance ranking as ``ts_rank``.
    Cached so the index is built once per ``path``.
    """
    df = load_icd10_codes(path)
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE VIRTUAL TABLE icd10_fts USING fts5(icd10_code UNINDEXED, description)"
    )
    conn.executemany(
        "INSERT INTO icd10_fts (icd10_code, description) VALUES (?, ?)",
        df[["ICD10_Code", "Description"]].itertuples(index=False, name=None),
    )
    return conn


def search_icd10_fts(
    query: str,
    limit: int = 20,
    path: str = str(ICD10_PARQUET),
) -> list[dict]:
    """Full-text search ICD-10-CM codes using SQLite FTS5 + BM25 ranking.

    Args:
        query: Free-text diagnosis name, e.g. "type 2 diabetes".
        limit: Max results to return.
        path: Path to the ICD-10 code parquet.

    Returns:
        List of match dicts (icd10_code, description, score), best match
        first. "score" is BM25 relevance (higher is better) — not on the same
        0-100 scale as ``search_icd10``'s fuzzy score, so the two aren't
        directly comparable numerically. Empty list if the query has no
        tokens or none of them match any description.
    """
    tokens = _FTS_TOKEN_RE.findall(query.lower())
    if not tokens:
        return []

    # Quote each token so it's matched literally, sidestepping FTS5's
    # reserved-character query syntax (e.g. a query containing "-" or "(").
    match_query = " ".join(f'"{token}"' for token in tokens)

    conn = _build_fts5_index(path)
    rows = conn.execute(
        """
        SELECT icd10_code, description, bm25(icd10_fts) AS rank
        FROM icd10_fts
        WHERE icd10_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (match_query, limit),
    ).fetchall()

    return [
        {"icd10_code": code, "description": description, "score": round(-rank, 2)}
        for code, description, rank in rows
    ]


def search_icd10_hybrid(
    query: str,
    limit: int = 20,
    pool: int = 50,
    rrf_k: int = 60,
    path: str = str(ICD10_PARQUET),
    scorer=_SCORER,
) -> list[dict]:
    """Hybrid ICD-10-CM search for a single diagnosis name.

    See ``search_icd10_hybrid_batch`` for how candidates are fused.
    """
    # Delegate to the batch path so single and batch rank identically, same
    # convention as search_icd10 / search_icd10_batch.
    return search_icd10_hybrid_batch(
        [query],
        limit=limit,
        pool=pool,
        rrf_k=rrf_k,
        path=path,
        scorer=scorer,
    )[query]


def search_icd10_hybrid_batch(
    queries: list[str],
    limit: int = 20,
    pool: int = 50,
    rrf_k: int = 60,
    path: str = str(ICD10_PARQUET),
    scorer=_SCORER,
) -> dict[str, list[dict]]:
    """Hybrid ICD-10-CM search: fuse FTS and fuzzy candidates via Reciprocal Rank Fusion.

    Runs both ``search_icd10_fts`` and ``search_icd10_batch`` over a wider
    candidate pool, then combines their rankings per query with RRF
    (``score = sum(1 / (rrf_k + rank))`` across whichever list(s) a code
    appears in). This returns close to the union of what either method alone
    finds, so for any given query it is rarely worse than the better
    individual method — it inherits FTS's stronger top-of-list relevance
    ranking where vocabulary overlaps, and fuzzy's ability to surface distant
    matches that share no literal terms with the query.

    Args:
        queries: Free-text diagnosis names, e.g. ["type 2 diabetes", "afib"].
        limit: Max results to return per query.
        pool: How many candidates to pull from each underlying method before fusing.
        rrf_k: RRF rank-damping constant (60 is the standard default).
        path: Path to the ICD-10 code parquet.
        scorer: rapidfuzz scorer passed through to the fuzzy half of the fuse
            (see search_icd10_batch).

    Returns:
        Mapping of each input query to its list of match dicts (same shape as
        ``search_icd10_hybrid``).
    """
    fuzzy_batch = search_icd10_batch(queries, limit=pool, score_cutoff=0, path=path, scorer=scorer)

    results: dict[str, list[dict]] = {}
    for query in queries:
        fts_results = search_icd10_fts(query, limit=pool, path=path)
        fuzzy_results = fuzzy_batch[query]

        fts_rank = {item["icd10_code"]: i for i, item in enumerate(fts_results, start=1)}
        fuzzy_rank = {
            item["icd10_code"]: i for i, item in enumerate(fuzzy_results, start=1)
        }
        by_code = {item["icd10_code"]: item for item in fts_results + fuzzy_results}

        fused = []
        for code, item in by_code.items():
            rrf_score = 0.0
            if code in fts_rank:
                rrf_score += 1.0 / (rrf_k + fts_rank[code])
            if code in fuzzy_rank:
                rrf_score += 1.0 / (rrf_k + fuzzy_rank[code])
            fused.append({**item, "score": round(rrf_score, 5)})

        fused.sort(key=lambda item: -item["score"])
        results[query] = fused[:limit]

    return results


if __name__ == "__main__":
    import json

    codes = load_icd10_codes()
    print(f"ICD-10-CM codes: {len(codes)}")

    print("\n== single search: 'community acquired pneumonia' ==")
    print(json.dumps(search_icd10("community acquired pneumonia", limit=5), indent=2))

    print("\n== batch search ==")
    batch = search_icd10_batch(
        ["acute bronchitis", "type 2 diabetes", "essential hypertension"],
        limit=3,
    )
    print(json.dumps(batch, indent=2))

    print("\n== FTS search: 'community acquired pneumonia' ==")
    print(json.dumps(search_icd10_fts("community acquired pneumonia", limit=5), indent=2))

    print("\n== hybrid (RRF) search: 'community acquired pneumonia' ==")
    print(json.dumps(search_icd10_hybrid("community acquired pneumonia", limit=5), indent=2))

    print("\n== hybrid (RRF) batch search ==")
    hybrid_batch = search_icd10_hybrid_batch(
        ["acute bronchitis", "type 2 diabetes", "essential hypertension"],
        limit=3,
    )
    print(json.dumps(hybrid_batch, indent=2))
