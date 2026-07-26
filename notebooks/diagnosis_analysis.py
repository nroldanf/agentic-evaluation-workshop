import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path

    return Path, mo, pd


@app.cell
def _(mo):
    mo.md("""
    # Diagnosis Parquet — Table Structure Analysis

    Loads `data/Diagnosis.parquet` with pandas (pyarrow engine) and
    inspects the structure of the table.
    """)
    return


@app.cell
def _(Path, pd):
    parquet_path = Path("../data") / "ICD10_DB.parquet"
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    df
    return (df,)


@app.cell
def _(df, mo):
    mo.md(f"""
    ## Shape

    - **Rows:** {df.shape[0]:,}
    - **Columns:** {df.shape[1]:,}
    - **Memory:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Column types & non-null counts
    """)
    return


@app.cell
def _(df, pd):
    schema = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "nulls": df.isna().sum(),
            "null_pct": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(),
        }
    )
    schema.index.name = "column"
    schema.reset_index()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Numeric summary
    """)
    return


@app.cell
def _(df):
    df.describe(include="number").T
    return


@app.cell
def _(mo):
    mo.md("""
    ## String / categorical summary
    """)
    return


@app.cell
def _(df):
    df.describe(include=["str", "category"]).T
    return


@app.cell
def _(mo):
    mo.md("""
    ## Datetime summary
    """)
    return


@app.cell
def _(df):
    df.describe(include=["datetime"]).T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Remove duplicate rows

    Rows that be safely deleted:
    - ICDCodeType different from 10
    - IsValid is False OR Valid is N
    - Where Exp_Date is not 1900-01-01 00:00:00 (this is a sentinel date)
    - Where Eft_Date is 1900-01-01 00:00:00 (this is a sentinel date)

    Columns that can be safely dropped:
    - Valid, IsValid, ICDCodeType, Short_Desc, Medium_Desc, Last_Upd_DateTime, DiagnosisCodePK, Last_Upd_UserID, Exp_Date, Eft_Date
    """)
    return


@app.cell
def _(df):
    simple_df = df.copy()
    # Remove ICDCodeType different from 10
    simple_df["ICDCodeType"] = simple_df["ICDCodeType"].astype(str)
    simple_df = simple_df[simple_df["ICDCodeType"] == "10"]
    # Remove non valid rows
    simple_df = simple_df[simple_df["IsValid"]]
    simple_df = simple_df[simple_df["Valid"] == "Y"]
    # # Keep Exp_Date with sentinel
    simple_df = simple_df[simple_df["Exp_Date"] == "1900-01-01 00:00:00"]
    # # Keep Eft_Date different from sentinel as long as Exp_Date is not sentinel
    # # Keep in mind that these are datetime columns, so we need to compare them as strings
    simple_df = simple_df[
    	(simple_df["Eft_Date"] != "1900-01-01 00:00:00")
    	| (simple_df["Exp_Date"] == "1900-01-01 00:00:00")
    ]
    return (simple_df,)


@app.cell
def _(simple_df):
    # Group by Diag_Code and count duplicates
    dupe_mask = simple_df["Diag_Code"].duplicated(keep=False)
    dupes = simple_df[dupe_mask].sort_values("Diag_Code")
    dupes
    return


@app.cell
def _(simple_df):
    deduped_df = (
        simple_df
        .assign(_desc_len=simple_df["Long_Desc"].str.len())
        .sort_values("_desc_len", ascending=False)
        .drop_duplicates(subset="Diag_Code", keep="first")
        .drop(columns="_desc_len")
        .sort_index()
    	.reset_index(drop=True)
    )
    deduped_df
    return (deduped_df,)


@app.cell
def _(deduped_df):
    # Drown unnecessary columns (Valid, IsValid, ICDCodeType, Short_Desc, Medium_Desc, Last_Upd_DateTime, DiagnosisCodePK, Last_Upd_UserID, Exp_Date, Eft_Date)
    final_df = deduped_df.drop(columns=["Valid", "IsValid", "ICDCodeType", "Short_Desc", "Medium_Desc", "Last_Upd_DateTime", "DiagnosisCodePK", "Last_Upd_UserID", "Exp_Date", "Eft_Date"])
    # Rename Diag_Code to ICD10_Code and Long_Desc to Description
    final_df = final_df.rename(columns={"Diag_Code": "ICD10_Code", "Long_Desc": "Description"})
    return (final_df,)


@app.cell
def _(Path, final_df):
    final_df.to_parquet(Path("../data") / "ICD10_DB.parquet", engine="pyarrow", index=False)
    return


@app.cell
def _(Path, pd):
    new_db = pd.read_parquet(Path("../data") / "ICD10_DB.parquet", engine="pyarrow")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tool for fuzzy search
    """)
    return


@app.cell
def _(Path):
    import sys

    repo_root = Path("..").resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from icd10_search import (
        load_icd10_codes,
        search_icd10,
        search_icd10_batch,
        search_icd10_fts,
        search_icd10_hybrid,
    )

    icd10_path = str(repo_root / "data" / "ICD10_DB.parquet")
    return (
        icd10_path,
        load_icd10_codes,
        search_icd10,
        search_icd10_batch,
        search_icd10_fts,
        search_icd10_hybrid,
    )


@app.cell
def _(icd10_path, load_icd10_codes):
    load_icd10_codes(icd10_path).shape
    return


@app.cell
def _(mo):
    mo.md("""
    ### Single-query search
    """)
    return


@app.cell
def _(icd10_path, search_icd10):
    search_icd10("pneumonia", limit=10, path=icd10_path)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Batch search
    """)
    return


@app.cell
def _(icd10_path, search_icd10_batch):
    search_icd10_batch(
        ["acute bronchitis", "type 2 diabetes", "essential hypertension"],
        limit=3,
        path=icd10_path,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Full-text search (SQLite FTS5 + BM25) — comparison

    In-memory analog of the Postgres `tsvector`/`plainto_tsquery` + `ts_rank`
    search: query terms are ANDed together (all must appear, in any order),
    ranked by BM25 relevance. Unlike the fuzzy search above, it requires exact
    token matches — so it can return nothing where the fuzzy search still
    finds a plausible neighbor (e.g. "acute bronchitis" has no ICD-10
    description containing both words together).
    """)
    return


@app.cell
def _(icd10_path, search_icd10_fts):
    search_icd10_fts("pneumonia unspecified", limit=10, path=icd10_path)
    return


@app.cell
def _(icd10_path, search_icd10_fts):
    {
        query: search_icd10_fts(query, limit=10, path=icd10_path)
        for query in ["bronchitis", "type 2 diabetes", "essential hypertension"]
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hybrid search (FTS + fuzzy, fused via Reciprocal Rank Fusion)

    Runs both underlying methods over a wider candidate pool and fuses their
    rankings (`score = sum(1 / (rrf_k + rank))` across whichever list(s) a
    code appears in), rather than picking one or the other. This returns
    close to the union of what either method finds, so for any given query
    it's rarely worse than the better individual method.
    """)
    return


@app.cell
def _(icd10_path, search_icd10_hybrid):
    search_icd10_hybrid("community acquired pneumonia", limit=10, path=icd10_path)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Quantitative comparison: fuzzy vs FTS vs hybrid

    25 diagnosis queries — a mix of exact clinical phrasing and
    colloquial/abbreviated terms a clinician might actually say — each paired
    with a hand-verified expected ICD-10 code (the code whose *exact*
    description in `ICD10_DB.parquet` matches the canonical diagnosis, e.g.
    "Essential (primary) hypertension" -> `I10`). For each query, check
    whether that code shows up in each method's top-25 results, and at what
    rank.
    """)
    return


@app.cell
def _():
    test_cases = [
        ("community acquired pneumonia", "J18.9"),
        ("acute bronchitis", "J20.9"),
        ("type 2 diabetes", "E11.9"),
        ("essential hypertension", "I10"),
        ("urinary tract infection", "N39.0"),
        ("chronic kidney disease", "N18.9"),
        ("afib", "I48.91"),
        ("major depressive disorder", "F32.9"),
        ("generalized anxiety disorder", "F41.1"),
        ("acid reflux", "K21.9"),
        ("copd", "J44.9"),
        ("sepsis", "A41.9"),
        ("congestive heart failure", "I50.9"),
        ("migraine without aura", "G43.009"),
        ("seizure", "R56.9"),
        ("dehydration", "E86.9"),
        ("acute kidney injury", "N17.9"),
        ("covid", "U07.1"),
        ("shortness of breath", "R06.02"),
        ("chest pain", "R07.9"),
        ("abdominal pain", "R10.9"),
        ("fever", "R50.9"),
        ("nausea", "R11.0"),
        ("headache", "R51.9"),
        ("gout", "M10.9"),
    ]
    return (test_cases,)


@app.cell
def _(
    icd10_path,
    pd,
    search_icd10,
    search_icd10_fts,
    search_icd10_hybrid,
    test_cases,
):
    def _rank_of(results, code):
        for i, item in enumerate(results, start=1):
            if item["icd10_code"] == code:
                return i
        return None

    comparison_rows = []
    for query, expected in test_cases:
        fuzzy_results = search_icd10(query, limit=25, score_cutoff=0, path=icd10_path)
        fts_results = search_icd10_fts(query, limit=25, path=icd10_path)
        hybrid_results = search_icd10_hybrid(query, limit=25, path=icd10_path)
        comparison_rows.append(
            {
                "query": query,
                "expected": expected,
                "fuzzy_rank": _rank_of(fuzzy_results, expected),
                "fuzzy_top1": fuzzy_results[0]["icd10_code"] if fuzzy_results else None,
                "fts_rank": _rank_of(fts_results, expected),
                "fts_top1": fts_results[0]["icd10_code"] if fts_results else None,
                "hybrid_rank": _rank_of(hybrid_results, expected),
                "hybrid_top1": hybrid_results[0]["icd10_code"] if hybrid_results else None,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df
    return (comparison_df,)


@app.cell
def _(comparison_df, pd):
    def _hit_at(rank_col, k):
        ranks = comparison_df[rank_col]
        return (ranks.notna() & (ranks <= k)).mean()

    metrics_df = pd.DataFrame(
        {
            "fuzzy": [_hit_at("fuzzy_rank", k) for k in (1, 5, 10, 25)],
            "fts": [_hit_at("fts_rank", k) for k in (1, 5, 10, 25)],
            "hybrid": [_hit_at("hybrid_rank", k) for k in (1, 5, 10, 25)],
        },
        index=["hit@1", "hit@5", "hit@10", "hit@25"],
    )
    metrics_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A production bug found via Langfuse traces: qualifier-heavy queries return nonsense

    `agent.py`'s diagnoses node (with the ICD-10 tool enabled) was observed
    burning 8+ tool calls on a single transcript. Its Langfuse trace's
    `reasoning_content` shows the model noticing bad tool results and retrying
    with progressively simpler phrasing — e.g.
    `search_icd10_hybrid("compartment syndrome of arm")` returned **`Z56.1`
    "Change of job"** and **`F40.230` "Fear of blood"** ahead of anything about
    compartment syndrome.

    Root cause: `search_icd10_fts` requires every query token to match (bareword
    AND semantics), so a query this specific returns nothing — no ICD-10
    description literally contains "of arm". With FTS empty, the hybrid fuse
    falls back entirely on the fuzzy `WRatio` scorer, and one of `WRatio`'s
    internal components (`partial_token_set_ratio`) saturates near 100 whenever
    query and candidate share even a single short common token — here, just
    "of" — regardless of relevance. That produces a flat, non-discriminating
    ~85.5 score for both nonsense matches and the true code alike, so nothing
    distinguishes them and the top result is essentially noise.

    This matters specifically because `diagnoses_prompt_with_tool.txt` instructs
    the model to name candidates "with full specificity: laterality, body site,
    acuity" — and it searches the tool with that same specific phrasing.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Adversarial queries: over-specified diagnosis names from the actual trace

    Acceptable-code sets are derived directly from `ICD10_DB.parquet`
    descriptions (not hand-picked) — a "hit" means a description matching the
    intended condition (and site, where the query specifies one) exists in the
    results, same convention as the `test_cases` comparison above.
    """)
    return


@app.cell
def _(icd10_path, load_icd10_codes):
    _codes_df = load_icd10_codes(icd10_path)

    def codes_for(*must_contain):
        mask = _codes_df["Description"].str.contains(must_contain[0], case=False, na=False)
        for term in must_contain[1:]:
            mask &= _codes_df["Description"].str.contains(term, case=False, na=False)
        return set(_codes_df.loc[mask, "ICD10_Code"])

    return (codes_for,)


@app.cell
def _(codes_for):
    adversarial_cases = [
        (
            "compartment syndrome upper limb unspecified side",
            codes_for("compartment syndrome", "upper extremity"),
        ),
        ("necrotizing fasciitis of shoulder", codes_for("necrotizing fasciitis")),
        (
            "sepsis with septic shock unspecified organism",
            codes_for("sepsis", "shock") | codes_for("septic shock"),
        ),
        ("retained foreign body in soft tissue", codes_for("foreign body", "soft tissue")),
        ("residual foreign body in soft tissue", codes_for("foreign body", "soft tissue")),
        ("tetanus", codes_for("tetanus")),
        ("peripheral neuropathy unspecified", codes_for("polyneuropathy, unspecified")),
        ("severe sepsis", codes_for("severe sepsis")),
    ]
    return (adversarial_cases,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scorer bake-off: does swapping WRatio fix it?

    Compare the current fuzzy scorer (`fuzz.WRatio`) against a candidate
    (`fuzz.token_set_ratio`, which drops the `partial_ratio` component
    responsible for the stopword-saturation bug above) on the adversarial
    queries. The fix has to earn its place on the original 25 general queries
    too — a scorer swap that fixes adversarial cases by breaking general ones
    isn't a fix.
    """)
    return


@app.cell
def _(adversarial_cases, icd10_path, pd, search_icd10_hybrid):
    from rapidfuzz import fuzz

    SCORERS = {"WRatio (current)": fuzz.WRatio, "token_set_ratio (candidate)": fuzz.token_set_ratio}

    def _rank_of_any(results, codes):
        for i, item in enumerate(results, start=1):
            if item["icd10_code"] in codes:
                return i
        return None

    _adversarial_rows = []
    for _query, _codes in adversarial_cases:
        _row = {"query": _query}
        for _name, _scorer in SCORERS.items():
            _hy = search_icd10_hybrid(_query, limit=25, path=icd10_path, scorer=_scorer)
            _row[f"{_name} rank"] = _rank_of_any(_hy, _codes)
            _row[f"{_name} top1"] = _hy[0]["description"] if _hy else None
        _adversarial_rows.append(_row)

    adversarial_df = pd.DataFrame(_adversarial_rows)
    adversarial_df
    return SCORERS, adversarial_df


@app.cell
def _(SCORERS, adversarial_df, pd):
    def _hit_at(col, k):
        ranks = adversarial_df[col]
        return (ranks.notna() & (ranks <= k)).mean()

    adversarial_metrics_df = pd.DataFrame(
        {name: [_hit_at(f"{name} rank", k) for k in (1, 5, 25)] for name in SCORERS},
        index=["topic hit@1", "topic hit@5", "topic hit@25"],
    )
    adversarial_metrics_df
    return


@app.cell
def _(SCORERS, icd10_path, pd, search_icd10_hybrid, test_cases):
    def _rank_of(results, code):
        for i, item in enumerate(results, start=1):
            if item["icd10_code"] == code:
                return i
        return None

    _general_metrics = {}
    for _name, _scorer in SCORERS.items():
        _ranks = pd.Series(
            [
                _rank_of(search_icd10_hybrid(q, limit=25, path=icd10_path, scorer=_scorer), c)
                for q, c in test_cases
            ]
        )
        _general_metrics[_name] = [(_ranks.notna() & (_ranks <= k)).mean() for k in (1, 5, 10, 25)]

    general_regression_df = pd.DataFrame(_general_metrics, index=["hit@1", "hit@5", "hit@10", "hit@25"])
    general_regression_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The comparison above is through the hybrid path
    (`search_icd10_hybrid_batch`) — the one actually exposed to agents as
    `search_icd10_codes_batch`. Worth also checking the *pure fuzzy* path
    (`search_icd10_batch`) in isolation, since `token_set_ratio` is known to
    saturate at 100 whenever the query's tokens are a subset of a candidate's
    (that's expected — e.g. "essential hypertension" ties every
    pregnancy-complicating variant at 100, and the shorter-description
    tiebreak in `search_icd10_batch` picks the general code, `I10`, correctly).
    Without FTS's independent ranking to fuse against, does that subset-tie
    behavior cost anything on the general queries?
    """)
    return


@app.cell
def _(SCORERS, icd10_path, pd, search_icd10, test_cases):
    def _rank_of_fuzzy(results, code):
        for i, item in enumerate(results, start=1):
            if item["icd10_code"] == code:
                return i
        return None

    _fuzzy_only_metrics = {}
    for _name, _scorer in SCORERS.items():
        _ranks = pd.Series(
            [
                _rank_of_fuzzy(
                    search_icd10(q, limit=25, score_cutoff=0, path=icd10_path, scorer=_scorer), c
                )
                for q, c in test_cases
            ]
        )
        _fuzzy_only_metrics[_name] = [(_ranks.notna() & (_ranks <= k)).mean() for k in (1, 5, 10, 25)]

    fuzzy_only_regression_df = pd.DataFrame(
        _fuzzy_only_metrics, index=["hit@1", "hit@5", "hit@10", "hit@25"]
    )
    fuzzy_only_regression_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Verdict

    `token_set_ratio` wins the adversarial set outright (the FOXG1/"Change of
    job" nonsense disappears, topic hit@5 rises noticeably) and, through the
    hybrid path, is never worse — slightly better, in fact — on the original 25
    general queries (hit@10 and hit@25 both improve, hit@1/hit@5 unchanged).
    Through the *pure fuzzy* path alone (no FTS to fuse against), it's a wash:
    one general query (`migraine without aura`) drops out of the top 5,
    offset by others moving closer to (but not into) the top 5. Since the only
    path actually exposed to agents is the hybrid one, this doesn't reach
    production — but it's the honest reason `token_set_ratio` isn't a
    strictly-dominant scorer, just a net-positive one where it matters. Switch
    `icd10_search.py`'s default `_SCORER` from `fuzz.WRatio` to
    `fuzz.token_set_ratio`.

    One adversarial case (`peripheral neuropathy unspecified`) misses under
    both scorers — a genuine vocabulary gap ("peripheral" isn't literally in
    "polyneuropathy"), not something a scorer swap fixes. That's a documented
    limitation, not a regression.

    This alone won't fully close the gap, though: `search_icd10_fts` still
    returns nothing for laterality/site/encounter-qualified queries regardless
    of scorer (its AND-all-tokens requirement isn't affected by the fuzzy
    component at all). The complementary fix is in
    `prompts/diagnoses_prompt_with_tool.txt`: search with the core condition
    term only, and apply the laterality/site specificity only when *selecting*
    among the codes returned — not in the query text itself.
    """)
    return


if __name__ == "__main__":
    app.run()
