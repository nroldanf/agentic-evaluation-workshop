import marimo

__generated_with = "0.23.9"
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
    parquet_path = Path("../data") / "Diagnosis.parquet"
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
def _(icd10_path, pd, search_icd10, search_icd10_fts, search_icd10_hybrid, test_cases):
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


if __name__ == "__main__":
    app.run()
