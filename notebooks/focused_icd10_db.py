import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import sys
    from pathlib import Path

    import httpx
    import marimo as mo
    import pandas as pd
    from langchain_ollama import ChatOllama

    repo_root = Path("..").resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from icd10_search import load_icd10_codes, search_icd10_hybrid, search_icd10_hybrid_batch

    icd10_path = str(repo_root / "data" / "ICD10_DB.parquet")
    golden_dir = repo_root / "data" / "golden"
    focused_path = repo_root / "data" / "ICD10_DB_focused.parquet"
    family_only_path = repo_root / "data" / "ICD10_DB_family_only.parquet"

    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))
    return (
        ChatOllama,
        OLLAMA_MODEL,
        OLLAMA_TIMEOUT,
        family_only_path,
        focused_path,
        golden_dir,
        httpx,
        icd10_path,
        json,
        load_icd10_codes,
        mo,
        pd,
        repo_root,
        search_icd10_hybrid,
        search_icd10_hybrid_batch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A focused ICD-10-CM database from the golden cases

    `data/ICD10_DB.parquet` covers every ICD-10-CM code. This notebook derives a
    much smaller subset, scoped to the code families and clinical terms that
    actually appear in `data/golden/`, and compares fuzzy/hybrid search quality
    on the full database against the focused one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the golden cases
    """)
    return


@app.cell
def _(golden_dir, json):
    golden_cases = []
    for _path in sorted(golden_dir.glob("*.json")):
        golden_cases.append({"source": _path.stem, **json.loads(_path.read_text(encoding="utf-8"))})
    return (golden_cases,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diagnoses referenced across differentials and assessments
    """)
    return


@app.cell
def _(golden_cases, pd):
    def _diagnosis_name(entry):
        return entry.get("diagnosis", entry.get("diagnosis_name"))

    diagnosis_rows = []
    for _case in golden_cases:
        for _section in ("differential_diagnoses", "assessment"):
            for _entry in _case.get(_section, []):
                diagnosis_rows.append(
                    {
                        "source": _case["source"],
                        "section": _section,
                        "icd10_code": _entry["icd10_code"],
                        "diagnosis_name": _diagnosis_name(_entry),
                    }
                )

    diagnoses_df = pd.DataFrame(diagnosis_rows)
    diagnoses_df
    return (diagnoses_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Search terms from the expected tool-call fixtures
    """)
    return


@app.cell
def _(golden_cases):
    forbidden_terms = set()
    query_terms = set()
    for _case in golden_cases:
        for _call in _case.get("expected_tool_calls", []):
            if _call.get("tool") != "lookup_icd10":
                continue
            query_terms.update(_call.get("expected_query_contains", []))
            forbidden_terms.update(_call.get("forbidden_query_contains", []))

    query_terms -= forbidden_terms
    sorted(query_terms)
    return (query_terms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Code families referenced by the golden diagnoses
    """)
    return


@app.cell
def _(diagnoses_df):
    golden_families = set(diagnoses_df["icd10_code"].str.split(".").str[0])
    sorted(golden_families)
    return (golden_families,)


@app.cell
def _(diagnoses_df, golden_families, icd10_path, load_icd10_codes):
    icd10_df = load_icd10_codes(icd10_path)
    _category = icd10_df["ICD10_Code"].str.split(".").str[0]

    family_only_codes = (
        set(icd10_df.loc[_category.isin(golden_families), "ICD10_Code"])
        | set(diagnoses_df["icd10_code"])
    )
    family_only_df = icd10_df[icd10_df["ICD10_Code"].isin(family_only_codes)].reset_index(drop=True)
    family_only_df
    return family_only_codes, family_only_df, icd10_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Differential terms suggested by the transcripts

    `golden_families` only covers the code categories of the diagnoses that
    made it into each case's final `differential_diagnoses`/`assessment`. A
    live diagnostic agent also has to search for conditions it reasonably
    considers and rules out along the way — those never appear in the golden
    codes at all, so `family_only` has no coverage for them. This asks a model
    to brainstorm that wider differential directly from each case's transcript,
    purely to expand the code pool, not to second-guess the golden diagnoses.
    """)
    return


@app.cell
def _(repo_root):
    transcript_map = {
        "RIV-001": repo_root / "data" / "encounter_riv001.txt",
        "HOU-002": repo_root / "data" / "encounter_hou002.txt",
    }

    def case_transcript(case):
        _path = transcript_map.get(case.get("encounter_id"))
        if _path is not None and _path.exists():
            return _path.read_text(encoding="utf-8")
        _parts = [case.get("hpi", "")]
        _parts.extend(_pe.get("findings", "") for _pe in case.get("physical_exam", []))
        return "\n".join(_parts)

    return (case_transcript,)


@app.cell
def _(ChatOllama, OLLAMA_MODEL, OLLAMA_TIMEOUT, httpx):
    term_extraction_model = ChatOllama(
        model=OLLAMA_MODEL,
        num_ctx=20480,
        keep_alive="15m",
        validate_model_on_init=True,
        temperature=0.1,
        reasoning=False,
        num_predict=512,
        client_kwargs={
            "timeout": httpx.Timeout(connect=10.0, read=OLLAMA_TIMEOUT, write=30.0, pool=10.0)
        },
    )
    return (term_extraction_model,)


@app.cell
def _(OLLAMA_MODEL, mo):
    mo.callout(
        mo.md(
            f"This cell makes a **live call** to the local Ollama model "
            f"`{OLLAMA_MODEL}` for every golden case — no cached fallback."
        ),
        kind="warn",
    )
    return


@app.cell
def _(case_transcript, golden_cases, term_extraction_model):
    _prompt = (
        "Read the clinical transcript below and list every condition a clinician "
        "might reasonably consider as a differential diagnosis, including less "
        "likely or ruled-out possibilities. One concise clinical term per line, no "
        "numbering, no ICD-10 codes, no explanations.\n\n<transcript>\n{transcript}\n</transcript>"
    )

    transcript_terms = set()
    for _case in golden_cases:
        _response = term_extraction_model.invoke(
            _prompt.format(transcript=case_transcript(_case))
        )
        transcript_terms.update(
            _line.strip("-* \t") for _line in _response.content.splitlines() if _line.strip()
        )

    len(transcript_terms)
    return (transcript_terms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Expand via search: pull in topically related codes
    """)
    return


@app.cell
def _(diagnoses_df, icd10_path, query_terms, search_icd10_hybrid_batch, transcript_terms):
    expansion_queries = sorted(set(diagnoses_df["diagnosis_name"]) | query_terms | transcript_terms)
    expansion_results = search_icd10_hybrid_batch(expansion_queries, limit=30, path=icd10_path)
    expansion_codes = {
        item["icd10_code"] for results in expansion_results.values() for item in results
    }
    len(expansion_codes)
    return (expansion_codes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Assemble the focused database
    """)
    return


@app.cell
def _(expansion_codes, family_only_codes, icd10_df):
    focused_codes = family_only_codes | expansion_codes
    focused_df = icd10_df[icd10_df["ICD10_Code"].isin(focused_codes)].reset_index(drop=True)
    focused_df
    return (focused_df,)


@app.cell
def _(focused_df, icd10_path, load_icd10_codes, mo):
    _full_n = len(load_icd10_codes(icd10_path))
    _focused_n = len(focused_df)
    mo.md(f"""
    - **Full database:** {_full_n:,} codes
    - **Focused database:** {_focused_n:,} codes
    - **Reduction:** {(1 - _focused_n / _full_n) * 100:.1f}%
    """)
    return


@app.cell
def _(family_only_df, family_only_path, focused_df, focused_path):
    family_only_df.to_parquet(family_only_path, engine="pyarrow", index=False)
    focused_df.to_parquet(focused_path, engine="pyarrow", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Search quality: full database vs. family-only vs. focused

    `family_only` is derived purely from ICD-10 code structure (the category
    prefixes above), so querying it with the golden diagnosis names is an
    independent test — nothing about those queries influenced which codes are
    in the pool. `focused` additionally includes codes pulled in by searching
    with these same diagnosis names, so its ranks on this exact query set are
    not an independent test; it's shown for completeness alongside its size.
    """)
    return


@app.cell
def _(
    diagnoses_df,
    family_only_path,
    focused_path,
    icd10_path,
    pd,
    search_icd10_hybrid,
):
    def _rank_of(results, code):
        for i, item in enumerate(results, start=1):
            if item["icd10_code"] == code:
                return i
        return None

    comparison_rows = []
    for _, _row in diagnoses_df.iterrows():
        _full = search_icd10_hybrid(_row["diagnosis_name"], limit=25, path=icd10_path)
        _family = search_icd10_hybrid(_row["diagnosis_name"], limit=25, path=str(family_only_path))
        _focused = search_icd10_hybrid(_row["diagnosis_name"], limit=25, path=str(focused_path))
        comparison_rows.append(
            {
                "diagnosis_name": _row["diagnosis_name"],
                "expected": _row["icd10_code"],
                "full_rank": _rank_of(_full, _row["icd10_code"]),
                "full_top1": _full[0]["description"] if _full else None,
                "family_only_rank": _rank_of(_family, _row["icd10_code"]),
                "family_only_top1": _family[0]["description"] if _family else None,
                "focused_rank": _rank_of(_focused, _row["icd10_code"]),
                "focused_top1": _focused[0]["description"] if _focused else None,
            }
        )

    golden_comparison_df = pd.DataFrame(comparison_rows)
    golden_comparison_df
    return (golden_comparison_df,)


@app.cell
def _(golden_comparison_df, pd):
    def _hit_at(rank_col, k):
        ranks = golden_comparison_df[rank_col]
        return (ranks.notna() & (ranks <= k)).mean()

    golden_metrics_df = pd.DataFrame(
        {
            "full": [_hit_at("full_rank", k) for k in (1, 3, 5)],
            "family_only": [_hit_at("family_only_rank", k) for k in (1, 3, 5)],
            "focused": [_hit_at("focused_rank", k) for k in (1, 3, 5)],
        },
        index=["hit@1", "hit@3", "hit@5"],
    )
    golden_metrics_df
    return


if __name__ == "__main__":
    app.run()
