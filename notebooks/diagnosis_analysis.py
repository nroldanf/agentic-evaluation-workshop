import marimo

__generated_with = "0.23.13"
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
    parquet_path = Path("data") / "Diagnosis.parquet"
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
