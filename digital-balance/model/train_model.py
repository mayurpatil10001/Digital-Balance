"""
train_model.py -- Model training, evaluation, hyperparameter tuning for Digital Balance.
Run: python model/train_model.py  (from the digital-balance/ root)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ── Resolve paths ──────────────────────────────────────────────────────────────
ROOT_DIR  = os.path.dirname(os.path.abspath(__file__))          # digital-balance/model/
BASE_DIR  = os.path.dirname(ROOT_DIR)                           # digital-balance/
DATA_PATH = os.path.join(BASE_DIR, "data", "ScreenTime_vs_MentalWellness.csv")
MODEL_OUT = os.path.join(ROOT_DIR, "wellness_model.pkl")
METRICS_OUT      = os.path.join(ROOT_DIR, "metrics.json")
FEAT_IMP_OUT     = os.path.join(ROOT_DIR, "feature_importance.json")
PLOTS_DIR        = os.path.join(ROOT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Add model dir to path so we can import preprocessing
sys.path.insert(0, ROOT_DIR)
from preprocessing import (
    load_and_clean, run_eda, build_preprocessor,
    get_feature_lists, prepare_data, TARGET, CAT_COLS, NUM_COLS
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(name: str, pipeline, X_train, y_train, X_test, y_test) -> dict:
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
    pipeline.fit(X_train, y_train)
    preds_test  = pipeline.predict(X_test)
    preds_train = pipeline.predict(X_train)
    return {
        "model":    name,
        "test_r2":  round(float(r2_score(y_test, preds_test)), 4),
        "test_mae": round(float(mean_absolute_error(y_test, preds_test)), 4),
        "test_rmse":round(rmse(y_test, preds_test), 4),
        "train_r2": round(float(r2_score(y_train, preds_train)), 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std":  round(float(cv_scores.std()), 4),
    }


def get_feature_names(pipeline, num_features, cat_features):
    """Extract feature names after OneHotEncoding inside the ColumnTransformer."""
    ct = pipeline.named_steps["preprocessor"]
    ohe = ct.named_transformers_["cat"]
    ohe_names = ohe.get_feature_names_out(cat_features).tolist()
    return num_features + ohe_names


def extract_importances(pipeline, num_features, cat_features) -> dict:
    """Return {feature_name: importance} sorted descending."""
    reg = pipeline.named_steps["regressor"]
    feat_names = get_feature_names(pipeline, num_features, cat_features)

    if hasattr(reg, "feature_importances_"):
        importances = reg.feature_importances_
    elif hasattr(reg, "coef_"):
        importances = np.abs(reg.coef_)
    else:
        return {}

    paired = sorted(zip(feat_names, importances.tolist()), key=lambda x: -x[1])
    return {k: round(v, 6) for k, v in paired}


# ── Main training loop ─────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  Digital Balance — Model Training Pipeline")
    print("="*60)

    # 1. Load & clean data
    df = load_and_clean(DATA_PATH)

    # 2. EDA plots
    print("\n[1/7] Generating EDA plots …")
    run_eda(df)

    # 3. Prepare features / target
    X, y = prepare_data(df)
    num_features, cat_features = get_feature_lists(df)
    print(f"      Features: {num_features + cat_features}")
    print(f"      Target  : {TARGET}  (mean={y.mean():.2f}, median={y.median():.2f})")

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n[2/7] Split -> train={len(X_train)}, test={len(X_test)}")

    # 5. Compute target percentile thresholds FROM TRAINING DATA
    q25 = float(np.percentile(y_train, 25))
    q50 = float(np.percentile(y_train, 50))
    q75 = float(np.percentile(y_train, 75))
    print(f"      Percentile thresholds (train): Q25={q25:.2f}, Q50={q50:.2f}, Q75={q75:.2f}")

    # 6. Build candidate pipelines
    preprocessor = build_preprocessor(num_features, cat_features)

    candidates = {
        "Linear Regression":        LinearRegression(),
        "Ridge Regression":          Ridge(alpha=1.0),
        "Random Forest":             RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":         GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = []
    fitted_pipelines = {}
    print("\n[3/7] Evaluating baseline models …")
    for name, regressor in candidates.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(num_features, cat_features)),
            ("regressor",    regressor),
        ])
        metrics = evaluate(name, pipe, X_train, y_train, X_test, y_test)
        results.append(metrics)
        fitted_pipelines[name] = pipe
        print(f"      {name:<25} test_R²={metrics['test_r2']:.4f}  MAE={metrics['test_mae']:.4f}  RMSE={metrics['test_rmse']:.4f}  CV_R²={metrics['cv_r2_mean']:.4f}±{metrics['cv_r2_std']:.4f}")

    # 7. Select best model
    results_df = pd.DataFrame(results).sort_values(["test_r2", "test_rmse"], ascending=[False, True])
    best_row = results_df.iloc[0]
    best_name = best_row["model"]
    print(f"\n[4/7] Best baseline model: {best_name}  (test_R²={best_row['test_r2']:.4f})")

    # 8. Hyperparameter tuning via RandomizedSearchCV
    print(f"\n[5/7] Hyperparameter tuning ({best_name}) …")
    best_regressor_class = type(candidates[best_name])

    if "Forest" in best_name:
        param_dist = {
            "regressor__n_estimators":     [100, 200, 300, 400, 500],
            "regressor__max_depth":        [None, 5, 10, 15, 20, 25],
            "regressor__min_samples_leaf": [1, 2, 4, 6, 8],
            "regressor__min_samples_split":[2, 5, 10],
            "regressor__max_features":     ["sqrt", "log2", 0.5, 0.7],
        }
    elif "Gradient" in best_name:
        param_dist = {
            "regressor__n_estimators":  [100, 200, 300, 400],
            "regressor__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "regressor__max_depth":     [2, 3, 4, 5, 6],
            "regressor__subsample":     [0.6, 0.7, 0.8, 0.9, 1.0],
            "regressor__min_samples_leaf": [1, 2, 4],
        }
    elif "Ridge" in best_name:
        param_dist = {
            "regressor__alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        }
    else:
        param_dist = {}  # Linear Regression has no hyperparams

    # Rebuild fresh pipeline for tuning
    tuning_pipe = Pipeline([
        ("preprocessor", build_preprocessor(num_features, cat_features)),
        ("regressor",    best_regressor_class(random_state=42) if "Forest" in best_name or "Gradient" in best_name else best_regressor_class()),
    ])

    if param_dist:
        search = RandomizedSearchCV(
            tuning_pipe, param_dist, n_iter=20, cv=5,
            scoring="r2", n_jobs=-1, random_state=42, verbose=0
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        print(f"      Best params: {search.best_params_}")
    else:
        best_pipeline = tuning_pipe
        best_pipeline.fit(X_train, y_train)

    # 9. Re-evaluate tuned model
    train_r2_tuned = r2_score(y_train, best_pipeline.predict(X_train))
    test_r2_tuned  = r2_score(y_test,  best_pipeline.predict(X_test))
    test_mae_tuned = mean_absolute_error(y_test, best_pipeline.predict(X_test))
    test_rmse_tuned= rmse(y_test, best_pipeline.predict(X_test))
    print(f"\n[6/7] Tuned model results:")
    print(f"      Train R²={train_r2_tuned:.4f}  Test R²={test_r2_tuned:.4f}  MAE={test_mae_tuned:.4f}  RMSE={test_rmse_tuned:.4f}")
    overfit_gap = train_r2_tuned - test_r2_tuned
    if overfit_gap > 0.15:
        print(f"      [WARN] Overfit warning: train-test R2 gap = {overfit_gap:.4f} > 0.15")
    else:
        print(f"      [OK] Overfit check passed (gap={overfit_gap:.4f})")

    # 10. Feature importances
    feat_imp = extract_importances(best_pipeline, num_features, cat_features)
    top5 = list(feat_imp.items())[:5]

    # 11. Save model comparison plot
    print(f"\n[7/7] Saving artefacts …")
    fig, ax = plt.subplots(figsize=(9, 5))
    model_names = [r["model"] for r in results]
    r2_vals     = [r["test_r2"] for r in results]
    colors = ["#4CAF50" if n == best_name else "#78909C" for n in model_names]
    bars = ax.barh(model_names, r2_vals, color=colors, edgecolor="white", height=0.55)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
    ax.set_xlabel("Test R²", fontsize=11)
    ax.set_title("Model Comparison — Test R²", fontsize=13)
    ax.set_xlim(0, max(r2_vals) * 1.18)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 12. Save metrics.json
    metrics_payload = {
        "all_models":     results,
        "best_model":     best_name,
        "tuned_test_r2":  round(test_r2_tuned, 4),
        "tuned_test_mae": round(test_mae_tuned, 4),
        "tuned_test_rmse":round(test_rmse_tuned, 4),
        "tuned_train_r2": round(train_r2_tuned, 4),
        "category_thresholds": {
            "poor_below":      round(q25, 4),
            "fair_below":      round(q50, 4),
            "good_below":      round(q75, 4),
            "excellent_above": round(q75, 4),
        },
        "dataset_stats": {
            "n_train": int(len(X_train)),
            "n_test":  int(len(X_test)),
            "target_mean":   round(float(y.mean()), 4),
            "target_median": round(float(y.median()), 4),
            "target_std":    round(float(y.std()), 4),
            "target_q25":    round(q25, 4),
            "target_q50":    round(q50, 4),
            "target_q75":    round(q75, 4),
        },
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # 13. Save feature_importance.json
    with open(FEAT_IMP_OUT, "w") as f:
        json.dump(feat_imp, f, indent=2)

    # 14. Save final pipeline
    joblib.dump(best_pipeline, MODEL_OUT)
    print(f"      [OK] Model saved   -> {MODEL_OUT}")
    print(f"      [OK] Metrics saved -> {METRICS_OUT}")
    print(f"      [OK] Features saved -> {FEAT_IMP_OUT}")

    # 15. Console summary
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    print(f"  Best model   : {best_name}")
    print(f"  Test R²      : {test_r2_tuned:.4f}")
    print(f"  Test MAE     : {test_mae_tuned:.4f}")
    print(f"  Test RMSE    : {test_rmse_tuned:.4f}")
    print(f"  Category thresholds (from training quartiles):")
    print(f"    Poor      -> score < {q25:.2f}")
    print(f"    Fair      -> {q25:.2f} <= score < {q50:.2f}")
    print(f"    Good      -> {q50:.2f} <= score < {q75:.2f}")
    print(f"    Excellent -> score >= {q75:.2f}")
    print(f"\n  Top 5 features by importance:")
    for i, (feat, imp) in enumerate(top5, 1):
        print(f"    {i}. {feat:<40} {imp:.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
