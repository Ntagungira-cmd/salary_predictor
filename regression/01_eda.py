"""
01_eda.py — Exploratory Data Analysis for the Data Science Job Salaries dataset.

Loads regression/data/ds_salaries.csv, produces the required plots under
regression/figures/, and prints a short interpretation after each plot that
ties findings back to feature-engineering choices.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pipeline import DATA_PATH, FIGURES_DIR, EXPERIENCE_ORDINAL, map_job_title_to_family, map_location_to_region

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    print(f"Columns: {list(df.columns)}")
    print(df.describe(include="all").transpose().to_string())
    return df


def plot_salary_histogram(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["salary_in_usd"], bins=40, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("salary_in_usd")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of salary_in_usd")
    fig.tight_layout()
    out = FIGURES_DIR / "salary_histogram.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\n[Saved] {out}")
    skew = df["salary_in_usd"].skew()
    print(
        f"Interpretation: salary_in_usd is right-skewed (skew={skew:.2f}), with a long "
        "tail of six-figure+ outliers up to ~$450k. A linear model on the raw USD target "
        "will be pulled by those outliers, so MAE/RMSE will look large even when ranking "
        "is decent — prefer reporting both R^2 and MAE, and consider whether log-transform "
        "of the target would help in a follow-up. Feature engineering should keep robust "
        "metrics in mind rather than chasing R^2 alone."
    )


def plot_salary_by_experience(df: pd.DataFrame) -> None:
    order = ["EN", "MI", "SE", "EX"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="experience_level",
        y="salary_in_usd",
        order=order,
        ax=ax,
        palette="Blues",
    )
    ax.set_title("salary_in_usd by experience_level")
    fig.tight_layout()
    out = FIGURES_DIR / "salary_by_experience.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\n[Saved] {out}")
    medians = df.groupby("experience_level")["salary_in_usd"].median().reindex(order)
    print(
        "Interpretation: experience_level clearly separates salary bands "
        f"(medians EN={medians['EN']:,.0f}, MI={medians['MI']:,.0f}, "
        f"SE={medians['SE']:,.0f}, EX={medians['EX']:,.0f}). The monotonic rise "
        "EN < MI < SE < EX means this feature has a natural order, so ordinal encoding "
        "(0/1/2/3) is a better fit than one-hot — one-hot would treat levels as unrelated "
        "categories and waste dimensionality."
    )


def plot_salary_by_company_size(df: pd.DataFrame) -> None:
    order = ["S", "M", "L"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="company_size",
        y="salary_in_usd",
        order=order,
        ax=ax,
        palette="Greens",
    )
    ax.set_title("salary_in_usd by company_size")
    fig.tight_layout()
    out = FIGURES_DIR / "salary_by_company_size.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\n[Saved] {out}")
    print(
        "Interpretation: company_size shows salary differences (medium companies often "
        "pay more than large in this dataset, with small firms lower and more variable), "
        "but there is no strict numeric order that maps cleanly onto pay. Treat "
        "company_size as nominal and one-hot encode S/M/L rather than forcing an "
        "ordinal scale that the data does not support."
    )


def plot_avg_salary_top_titles(df: pd.DataFrame) -> None:
    top15 = df["job_title"].value_counts().head(15).index
    subset = df[df["job_title"].isin(top15)]
    avg = (
        subset.groupby("job_title")["salary_in_usd"]
        .mean()
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    avg.plot(kind="barh", ax=ax, color="#DD8452")
    ax.set_xlabel("Average salary_in_usd")
    ax.set_title("Average salary_in_usd for top-15 job titles (by count)")
    fig.tight_layout()
    out = FIGURES_DIR / "avg_salary_top15_titles.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\n[Saved] {out}")
    n_unique = df["job_title"].nunique()
    print(
        f"Interpretation: there are {n_unique} unique job titles; the top 15 dominate "
        "counts but salaries still vary a lot within and across titles (e.g. ML Engineer "
        "vs Data Analyst). One-hotting all 93 titles would explode dimensionality and "
        "leave rare titles almost unidentifiable. Collapse titles into ~6 role families "
        "(Leadership, ML/Research, Scientist, Engineer, Analyst, Other) via keyword "
        "mapping so the model sees stable, mentorship-relevant career tracks."
    )


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Build a small encoded frame (ordinal experience, remote_ratio, work_year,
    plus simple numeric codes for size/region) for a correlation heatmap.
    """
    enc = pd.DataFrame(
        {
            "salary_in_usd": df["salary_in_usd"],
            "work_year": df["work_year"],
            "remote_ratio": df["remote_ratio"],
            "experience_ord": df["experience_level"].map(EXPERIENCE_ORDINAL),
            "company_size_ord": df["company_size"].map({"S": 0, "M": 1, "L": 2}),
            "region_us": (df["company_location"] == "US").astype(int),
            "is_ft": (df["employment_type"] == "FT").astype(int),
        }
    )
    corr = enc.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap (numeric / lightly encoded features)")
    fig.tight_layout()
    out = FIGURES_DIR / "correlation_heatmap.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\n[Saved] {out}")
    print(
        "Interpretation: experience_ord and the US-location flag show the strongest "
        "positive correlations with salary_in_usd; remote_ratio and company_size_ord "
        "are weaker. That confirms we should keep ordinal experience and a region "
        "bucket (US vs Europe/Asia/Other) as first-class features, and that "
        "one-hotting every country would mostly add sparse noise. Weakly correlated "
        "features can still help interactively (e.g. remote × region) but a linear "
        "model may down-weight them — we will check coefficients after training."
    )


def main() -> None:
    df = load()
    plot_salary_histogram(df)
    plot_salary_by_experience(df)
    plot_salary_by_company_size(df)
    plot_avg_salary_top_titles(df)
    plot_correlation_heatmap(df)
    print("\nEDA complete. Figures are in:", FIGURES_DIR)


if __name__ == "__main__":
    main()
