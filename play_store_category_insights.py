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
    # Guard against NaNs in ratings; treat NaN as "unknown" and exclude from low-quality rule.
    s["avg_rating"] = pd.to_numeric(s["avg_rating"], errors="coerce")

    apps_q3 = s["num_apps"].quantile(0.75)
    apps_q1 = s["num_apps"].quantile(0.25)
    installs_per_app_median = s["installs_per_app"].median()
    installs_q3 = s["total_installs"].quantile(0.75)
    rating_median = s["avg_rating"].dropna().median() if s["avg_rating"].notna().any() else 0.0

    overcrowded = s[
        (s["num_apps"] >= apps_q3) & (s["installs_per_app"] <= installs_per_app_median)
    ].sort_values(["num_apps", "installs_per_app"], ascending=[False, True])

    underserved = s[
        (s["num_apps"] <= apps_q1) & (s["installs_per_app"] >= installs_per_app_median)
    ].sort_values(["installs_per_app", "total_installs"], ascending=[False, False])

    high_demand_low_quality = s[
        (s["total_installs"] >= installs_q3)
        & (s["avg_rating"].notna())
        & (s["avg_rating"] <= max(3.8, rating_median - 0.2))
    ].sort_values(["total_installs", "avg_rating"], ascending=[False, True])

    oc = _pick_first(overcrowded["Category"].tolist(), 2)
    un = _pick_first(underserved["Category"].tolist(), 2)
    hd = _pick_first(high_demand_low_quality["Category"].tolist(), 1)

    bullets: list[str] = []
    if oc:
        bullets.append(
            f"Overcrowded: {', '.join(oc)} have many apps but comparatively low installs per app."
        )
    if un:
        bullets.append(
            f"Underserved: {', '.join(un)} have few apps yet strong installs per app (room for entrants)."
        )
    if hd:
        bullets.append(
            f"High-demand but low-quality: {', '.join(hd)} combines high installs with below-par average ratings."
        )

    # Fill remaining bullets with highest-signal categories by simple heuristics.
    top_installs = s.sort_values("total_installs", ascending=False).head(1)["Category"].tolist()
    top_apps = s.sort_values("num_apps", ascending=False).head(1)["Category"].tolist()
    low_rating = (
        s[s["avg_rating"].notna()]
        .sort_values(["avg_rating", "total_installs"], ascending=[True, False])
        .head(1)["Category"]
        .tolist()
    )

    extras = [
        f"Demand concentration: {', '.join(top_installs)} leads total installs, signaling where users already are.",
        f"Competitive pressure: {', '.join(top_apps)} has the largest supply of apps, making differentiation harder.",
        f"Quality risk: {', '.join(low_rating)} shows the lowest average ratings (improvement opportunity).",
    ]
    for b in extras:
        if len(bullets) >= 5:
            break
        bullets.append(b)

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

