"""
preprocessing.py — Data cleaning, EDA, and preprocessing utilities for Digital Balance.
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "ScreenTime_vs_MentalWellness.csv")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Column definitions ─────────────────────────────────────────────────────────
TARGET      = "mental_wellness_index_0_100"
DROP_COLS   = ["user_id", "Unnamed: 15"]
CAT_COLS    = ["gender", "occupation", "work_mode"]
NUM_COLS    = [
    "age", "screen_time_hours", "work_screen_hours", "leisure_screen_hours",
    "sleep_hours", "sleep_quality_1_5", "stress_level_0_10",
    "productivity_0_100", "exercise_minutes_per_week", "social_hours_per_week",
]


# ── 1. Load & clean ────────────────────────────────────────────────────────────
def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """Load CSV, strip trailing column, validate ranges, cap impossible values."""
    df = pd.read_csv(path)

    # Drop trailing empty column
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # Strip whitespace from string columns
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── Missing value report ───────────────────────────────────────────────────
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"Missing values:\n{missing[missing > 0]}")
    else:
        logger.info("No missing values detected")

    # ── Duplicate user_ids ─────────────────────────────────────────────────────
    if "user_id" in df.columns:
        dups = df["user_id"].duplicated().sum()
        if dups:
            logger.warning(f"{dups} duplicate user_id(s) found")

    # ── Out-of-range validation ────────────────────────────────────────────────
    checks = {
        "age":                    (16, 100),
        "screen_time_hours":      (0, 24),
        "work_screen_hours":      (0, 24),
        "leisure_screen_hours":   (0, 24),
        "sleep_hours":            (0, 24),
        "sleep_quality_1_5":      (1, 5),
        "stress_level_0_10":      (0, 10),
        "productivity_0_100":     (0, 100),
        "exercise_minutes_per_week": (0, 10080),  # max minutes in a week
        "social_hours_per_week":  (0, 168),
        TARGET:                   (0, 100),
    }
    for col, (lo, hi) in checks.items():
        if col not in df.columns:
            continue
        n_neg = (df[col] < 0).sum()
        n_high = (df[col] > hi).sum()
        if n_neg:
            logger.warning(f"  {col}: {n_neg} rows with negative values → capping to {lo}")
            df[col] = df[col].clip(lower=lo)
        if n_high:
            logger.warning(f"  {col}: {n_high} rows exceeding {hi} → capping to {hi}")
            df[col] = df[col].clip(upper=hi)

    # ── Screen time consistency check ─────────────────────────────────────────
    if all(c in df.columns for c in ["screen_time_hours", "work_screen_hours", "leisure_screen_hours"]):
        discrepancy = (df["screen_time_hours"] - df["work_screen_hours"] - df["leisure_screen_hours"]).abs()
        flagged = (discrepancy > 1).sum()
        logger.info(f"Screen-time consistency: {flagged} rows where |total - work - leisure| > 1 hr (kept, not dropped)")

    logger.info(f"After cleaning: {len(df)} rows")
    return df


# ── 2. EDA plots ───────────────────────────────────────────────────────────────
PALETTE = "coolwarm"

def _save(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot: {path}")


def run_eda(df: pd.DataFrame):
    num_df = df[[c for c in NUM_COLS if c in df.columns] + [TARGET]]

    # -- Correlation heatmap ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Correlation Heatmap — Numeric Features vs Mental Wellness", fontsize=13, pad=12)
    _save(fig, "correlation_heatmap.png")

    # -- Screen time vs wellness -----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, xcol, label in zip(
        axes,
        ["screen_time_hours", "leisure_screen_hours"],
        ["Total Screen Time (hrs/day)", "Leisure Screen Time (hrs/day)"]
    ):
        if xcol in df.columns:
            ax.scatter(df[xcol], df[TARGET], alpha=0.45, c=df[TARGET],
                       cmap="RdYlGn", edgecolors="none", s=35)
            m, b = np.polyfit(df[xcol].dropna(), df.loc[df[xcol].notna(), TARGET], 1)
            xs = np.linspace(df[xcol].min(), df[xcol].max(), 200)
            ax.plot(xs, m * xs + b, "k--", linewidth=1.5, label=f"slope={m:.2f}")
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel("Mental Wellness Index", fontsize=11)
            ax.set_title(f"{label}\nvs Mental Wellness", fontsize=11)
            ax.legend(fontsize=9)
    fig.suptitle("Screen Time vs Mental Wellness", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "screentime_vs_wellness.png")

    # -- Stress vs sleep (colored by wellness) ---------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df["stress_level_0_10"], df["sleep_quality_1_5"],
        c=df[TARGET], cmap="RdYlGn", alpha=0.55, edgecolors="none", s=45
    )
    fig.colorbar(sc, ax=ax, label="Mental Wellness Index")
    ax.set_xlabel("Stress Level (0–10)", fontsize=11)
    ax.set_ylabel("Sleep Quality (1–5)", fontsize=11)
    ax.set_title("Stress Level vs Sleep Quality\n(colored by Mental Wellness Index)", fontsize=12)
    _save(fig, "stress_vs_sleep.png")

    logger.info("EDA plots complete")


# ── 3. Feature / target split & preprocessing ─────────────────────────────────
def build_preprocessor(num_features: list, cat_features: list) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
    )


def get_feature_lists(df: pd.DataFrame):
    """Return (num_features, cat_features) present in df (excluding target)."""
    num_feats = [c for c in NUM_COLS if c in df.columns]
    cat_feats  = [c for c in CAT_COLS if c in df.columns]
    return num_feats, cat_feats


def prepare_data(df: pd.DataFrame):
    """Return X (DataFrame), y (Series)."""
    feature_cols = [c for c in NUM_COLS + CAT_COLS if c in df.columns]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y


if __name__ == "__main__":
    df = load_and_clean()
    run_eda(df)
    X, y = prepare_data(df)
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Target stats:\n{y.describe()}")
