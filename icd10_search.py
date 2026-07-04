"""In-memory ICD-10-CM fuzzy search over ``data/Diagnosis.parquet``.

Why not PostgreSQL? The reference service (Experity/diagnosis-mcp) uses Postgres
with the ``pg_trgm`` extension for trigram fuzzy search. That requires a running
DB container and a load step. For a workshop tool the same job is done fully
in-memory with ``rapidfuzz`` over a pandas DataFrame — no Docker, no DDL.

The parquet is a slowly-changing-dimension (SCD-2) table: one code can have
several rows across effective/expiration windows. ``load_active_diagnoses``
collapses it to a deduplicated snapshot of current codes (see its docstring).

Scorer note: the reference ranks on ``word_similarity(Long_Desc, query)``, which
is *asymmetric* — it finds the best-matching token subset of the query inside the
(longer) description. We match against ``Long_Desc`` with ``rapidfuzz.fuzz.WRatio``
(a length-aware blend of ratio and token scorers): it tolerates the query being a
subset of a longer description without saturating at 100 for every over-specific
variant the way ``token_set_ratio`` does. On score ties we prefer the shorter
(more general) description, so e.g. "essential hypertension" ranks ``I10`` above
its pregnancy-complicating variants.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process, utils

DIAGNOSIS_PARQUET = Path("data/Diagnosis.parquet")

# Sentinel expiration date used in the source data: Exp_Date = 1900-01-01 means
# the code has no expiration, i.e. it is currently active.
SENTINEL_EXP_DATE = pd.Timestamp(1900, 1, 1)

# ICDCodeType 10 = ICD-10-CM (9 = legacy ICD-9-CM, which we exclude).
ICD10_CODE_TYPE = 10

# Default fuzzy-match scorer and preprocessing (lowercase, strip punctuation).
_SCORER = fuzz.WRatio
_PROCESSOR = utils.default_process


@lru_cache(maxsize=4)
def load_active_diagnoses(
    path: str = str(DIAGNOSIS_PARQUET),
    billable_only: bool = True,
) -> pd.DataFrame:
    """Load the SCD-2 parquet and collapse it to a deduplicated code snapshot.

    Simplification steps (SCD-2 -> current snapshot):
      1. Keep only ICD-10-CM rows (``ICDCodeType == 10``).
      2. If ``billable_only`` (default), keep only active, valid codes:
         ``IsValid == True`` AND ``Exp_Date == 1900-01-01`` (no expiration).
      3. Deduplicate to one row per ``Diag_Code``, keeping the most recently
         updated row (``Last_Upd_DateTime`` desc, then active rows first).

    Cached so the parquet is read and processed once per (path, billable_only).
    """
    df = pd.read_parquet(path)

    df = df[df["ICDCodeType"] == ICD10_CODE_TYPE]
    if billable_only:
        df = df[(df["IsValid"]) & (df["Exp_Date"] == SENTINEL_EXP_DATE)]

    # Prefer active (sentinel Exp_Date) and most-recently-updated rows, then dedup.
    df = df.assign(_active=df["Exp_Date"] == SENTINEL_EXP_DATE)
    df = df.sort_values(
        ["_active", "Last_Upd_DateTime"], ascending=[False, False]
    )
    df = df.drop_duplicates(subset="Diag_Code", keep="first")

    return df.reset_index(drop=True)


def _row_to_item(row: pd.Series) -> dict:
    """Shape a DataFrame row into the search-result item (code + description)."""
    return {
        "icd10_code": row["Diag_Code"],
        "description": row["Long_Desc"],
    }


def search_icd10(
    query: str,
    limit: int = 20,
    score_cutoff: float = 60.0,
    billable_only: bool = True,
    path: str = str(DIAGNOSIS_PARQUET),
) -> list[dict]:
    """Fuzzy-search ICD-10-CM codes for a single diagnosis name.

    Args:
        query: Free-text diagnosis name, e.g. "community acquired pneumonia".
        limit: Max results to return.
        score_cutoff: Minimum similarity score (0-100) to include a match.
        billable_only: Restrict to active, valid codes (default True).
        path: Path to the diagnosis parquet.

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
        billable_only=billable_only,
        path=path,
    )[query]


def search_icd10_batch(
    queries: list[str],
    limit: int = 20,
    score_cutoff: float = 60.0,
    billable_only: bool = True,
    path: str = str(DIAGNOSIS_PARQUET),
) -> dict[str, list[dict]]:
    """Fuzzy-search ICD-10-CM codes for many diagnosis names at once.

    Uses ``rapidfuzz.process.cdist`` to score all queries against all codes in
    one vectorized (multi-threaded) pass, then takes the top matches per query.

    Returns:
        Mapping of each input query to its list of match dicts (same shape as
        ``search_icd10``).
    """
    df = load_active_diagnoses(path, billable_only)
    choices = df["Long_Desc"].tolist()
    desc_len = df["Long_Desc"].str.len().to_numpy()

    # Score matrix: shape (len(queries), len(choices)).
    scores = process.cdist(
        queries,
        choices,
        scorer=_SCORER,
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


if __name__ == "__main__":
    import json

    active = load_active_diagnoses()
    print(f"Active ICD-10-CM codes: {len(active)}")

    print("\n== single search: 'community acquired pneumonia' ==")
    print(json.dumps(search_icd10("community acquired pneumonia", limit=5), indent=2))

    print("\n== batch search ==")
    batch = search_icd10_batch(
        ["acute bronchitis", "type 2 diabetes", "essential hypertension"],
        limit=3,
    )
    print(json.dumps(batch, indent=2))
