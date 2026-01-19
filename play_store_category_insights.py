from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class Outputs:
    summary_csv: Path
    apps_per_category_png: Path
    installs_per_category_png: Path


def _candidate_csv_paths(workspace_root: Path) -> list[Path]:
    # Common locations/names used for the Kaggle "Google Play Store Apps" dataset.
    candidates = [
        workspace_root / "googleplaystore.csv",
        workspace_root / "GooglePlayStore.csv",
        workspace_root / "data" / "googleplaystore.csv",
        workspace_root / "data" / "play_store" / "googleplaystore.csv",
        workspace_root / "data" / "raw" / "googleplaystore.csv",
    ]
    return [p for p in candidates if p.exists()]


def _download_from_kagglehub(dataset_slug: str = "lava18/google-play-store-apps") -> Path | None:
    """
    Attempts to download dataset via kagglehub (requires Kaggle credentials).
    Returns the path to the downloaded dataset directory if successful; otherwise None.
    """
    try:
        import kagglehub  # type: ignore
    except Exception:
        return None

    try:
        dataset_dir = Path(kagglehub.dataset_download(dataset_slug))
        return dataset_dir
    except Exception:
        return None


def load_google_play_store_df(workspace_root: Path) -> pd.DataFrame:
    """
    Load Google Play Store dataset into a DataFrame.

    - Prefers an existing local CSV in common locations.
    - Falls back to kagglehub download (requires Kaggle creds).
    """
    local = _candidate_csv_paths(workspace_root)
    if local:
        return pd.read_csv(local[0])

    dataset_dir = _download_from_kagglehub()
    if dataset_dir is None:
        raise FileNotFoundError(
            "Could not find 'googleplaystore.csv' locally, and kagglehub download failed. "
            "Place the CSV at the repo root (googleplaystore.csv) or in data/play_store/, "
            "or configure Kaggle credentials for kagglehub."
        )

    # Locate the CSV inside downloaded dataset folder.
    matches = list(dataset_dir.rglob("googleplaystore.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Downloaded Kaggle dataset to {dataset_dir}, but 'googleplaystore.csv' was not found."
        )

    return pd.read_csv(matches[0])


def _to_int_series(series: pd.Series) -> pd.Series:
    # Converts strings like "1,000+", "10+", "0" to integers; non-parsable -> 0
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
    )
    # Some datasets may have "Free" or other odd tokens; coerce to NaN then fill 0.
    return pd.to_numeric(cleaned, errors="coerce").fillna(0).astype("int64")


def _to_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Category", "Installs", "Rating", "Reviews"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Dataset missing required columns: {sorted(missing)}")

    data = df.copy()
    data["Category"] = data["Category"].astype(str).str.strip().fillna("Unknown")
    data["_installs"] = _to_int_series(data["Installs"])
    data["_reviews"] = _to_int_series(data["Reviews"])
    data["_rating"] = _to_float_series(data["Rating"])

    # Known data quality issues in common Play Store CSVs:
    # - A mis-shifted row can produce a numeric "Category" like "1.9" and an out-of-range rating.
    # - Ratings should be within [0, 5] when present.
    data = data[~data["Category"].astype(str).str.fullmatch(r"\d+(\.\d+)?", na=False)]
    data = data[data["_rating"].isna() | data["_rating"].between(0, 5)]

    app_col = "App" if "App" in data.columns else None
    count_col = (app_col, "size") if app_col else ("Category", "size")

    summary = (
        data.groupby("Category", dropna=False)
        .agg(
            num_apps=count_col,
            total_installs=("_installs", "sum"),
            avg_rating=("_rating", "mean"),
            total_reviews=("_reviews", "sum"),
        )
        .reset_index()
    )

    summary["installs_per_app"] = summary["total_installs"] / summary["num_apps"].clip(lower=1)
    summary["reviews_per_app"] = summary["total_reviews"] / summary["num_apps"].clip(lower=1)

    # Requested sort: installs desc, apps asc
    summary = summary.sort_values(["total_installs", "num_apps"], ascending=[False, True]).reset_index(
        drop=True
    )
    return summary


def plot_bar_charts(summary: pd.DataFrame, outputs: Outputs) -> None:
    outputs.apps_per_category_png.parent.mkdir(parents=True, exist_ok=True)

    # Apps per category: horizontal is more readable for many categories.
    by_apps = summary.sort_values("num_apps", ascending=True)
    plt.figure(figsize=(12, max(6, 0.32 * len(by_apps))))
    plt.barh(by_apps["Category"], by_apps["num_apps"])
    plt.title("Number of apps per category")
    plt.xlabel("Number of apps")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(outputs.apps_per_category_png, dpi=200)
    plt.close()

    # Total installs per category: also horizontal; sorted ascending for barh.
    by_installs = summary.sort_values("total_installs", ascending=True)
    plt.figure(figsize=(12, max(6, 0.32 * len(by_installs))))
    plt.barh(by_installs["Category"], by_installs["total_installs"])
    plt.title("Total installs per category")
    plt.xlabel("Total installs")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(outputs.installs_per_category_png, dpi=200)
    plt.close()


def _pick_first(items: Iterable[str], n: int) -> list[str]:
    out: list[str] = []
    for x in items:
        if x not in out:
            out.append(x)
        if len(out) >= n:
            break
    return out


def generate_insights(summary: pd.DataFrame) -> list[str]:
    """
    Returns 5 concise bullets covering:
    - overcrowded (many apps, low installs/app)
    - underserved (few apps, high installs/app)
    - high-demand but low-quality (high installs, low rating)
    """
    s = summary.copy()
    s["avg_rating"] = pd.to_numeric(s["avg_rating"], errors="coerce")

    # Overcrowded = lots of apps, relatively weak installs/app.
    apps_q3 = s["num_apps"].quantile(0.75)
    overcrowded_candidates = s[s["num_apps"] >= apps_q3].sort_values(
        ["installs_per_app", "num_apps"], ascending=[True, False]
    )
    oc = _pick_first(overcrowded_candidates["Category"].tolist(), 2)

    # Underserved = few apps, strong installs/app.
    apps_q1 = s["num_apps"].quantile(0.25)
    underserved_candidates = s[s["num_apps"] <= apps_q1].sort_values(
        ["installs_per_app", "total_installs"], ascending=[False, False]
    )
    un = _pick_first(underserved_candidates["Category"].tolist(), 2)

    # High-demand but low-quality = high installs, lowest rating within that high-demand set.
    installs_q3 = s["total_installs"].quantile(0.75)
    high_demand = s[(s["total_installs"] >= installs_q3) & (s["avg_rating"].notna())]
    hd = (
        high_demand.sort_values(["avg_rating", "total_installs"], ascending=[True, False])
        .head(1)["Category"]
        .tolist()
    )

    bullets: list[str] = []
    for cat in oc:
        row = s.loc[s["Category"] == cat].iloc[0]
        bullets.append(
            f"Overcrowded: {cat} has {int(row['num_apps'])} apps with only ~{row['installs_per_app']:.1e} installs/app."
        )
    for cat in un:
        row = s.loc[s["Category"] == cat].iloc[0]
        bullets.append(
            f"Underserved: {cat} has just {int(row['num_apps'])} apps but ~{row['installs_per_app']:.1e} installs/app."
        )
    if hd:
        cat = hd[0]
        row = s.loc[s["Category"] == cat].iloc[0]
        bullets.append(
            f"High-demand but low-quality: {cat} is top-quartile in installs yet averages only {row['avg_rating']:.2f} rating."
        )

    # Ensure exactly 5 bullets (pad with additional high-signal observations if needed).
    if len(bullets) < 5:
        top_installs = s.sort_values("total_installs", ascending=False).head(5)
        for _, r in top_installs.iterrows():
            if len(bullets) >= 5:
                break
            bullets.append(
                f"High demand: {r['Category']} contributes {int(r['total_installs'])} total installs across {int(r['num_apps'])} apps."
            )

    return bullets[:5]


def main() -> None:
    workspace_root = Path(os.environ.get("WORKSPACE_ROOT", Path.cwd()))
    outputs_dir = workspace_root / "outputs" / "play_store"
    outputs = Outputs(
        summary_csv=outputs_dir / "category_summary.csv",
        apps_per_category_png=outputs_dir / "apps_per_category.png",
        installs_per_category_png=outputs_dir / "installs_per_category.png",
    )

    df = load_google_play_store_df(workspace_root)
    summary = build_category_summary(df)

    # Save summary and charts
    outputs_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outputs.summary_csv, index=False)
    plot_bar_charts(summary, outputs)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 50)
    print("\nCategory summary (sorted by total installs desc, num apps asc):\n")
    print(summary.to_string(index=False))

    print("\nInsights:\n")
    for bullet in generate_insights(summary):
        print(f"- {bullet}")

    print("\nSaved outputs:")
    print(f"- {outputs.summary_csv}")
    print(f"- {outputs.apps_per_category_png}")
    print(f"- {outputs.installs_per_category_png}")


if __name__ == "__main__":
    main()

