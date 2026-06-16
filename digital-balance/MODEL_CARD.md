# Model Card — Digital Balance Mental Wellness Predictor

## Model Overview

| Field | Value |
|---|---|
| **Model name** | Digital Balance Wellness Predictor |
| **Version** | 1.0 |
| **Model type** | Regression (see Metrics section for final model type after training) |
| **Framework** | scikit-learn Pipeline |
| **Output** | Continuous score 0–100 (`mental_wellness_index_0_100`), binned post-hoc into Poor / Fair / Good / Excellent using training-set quartiles |
| **Intended use** | Educational demonstration of the statistical relationship between screen time, sleep, stress, and self-reported psychological well-being |
| **NOT intended for** | Clinical diagnosis, mental health screening, insurance, employment, or any consequential decision-making |

---

## Dataset

| Field | Details |
|---|---|
| **Source** | Simulated / academic survey-style dataset |
| **Size** | 400 rows × 15 feature columns + 1 target |
| **Collection method** | Self-reported survey responses |
| **Target variable** | `mental_wellness_index_0_100` — self-reported composite wellness score |
| **Target distribution** | Heavily right-skewed; mean ≈ 20, median ≈ 15; most participants cluster in the low range with relatively few high-scorers |

### Features

| Feature | Type | Range |
|---|---|---|
| `age` | Numeric | 16–60 |
| `gender` | Categorical | Female, Male, Non-binary/Other |
| `occupation` | Categorical | Employed, Student, Self-employed, Retired, Unemployed |
| `work_mode` | Categorical | Remote, In-person, Hybrid |
| `screen_time_hours` | Numeric | ~1–19 hrs/day |
| `work_screen_hours` | Numeric | ~0–12 hrs/day |
| `leisure_screen_hours` | Numeric | ~0–13 hrs/day |
| `sleep_hours` | Numeric | ~4–9 hrs/night |
| `sleep_quality_1_5` | Ordinal | 1–5 |
| `stress_level_0_10` | Numeric | 0–10 |
| `productivity_0_100` | Numeric | 0–100 |
| `exercise_minutes_per_week` | Numeric | 0–300+ |
| `social_hours_per_week` | Numeric | 0–24 |

---

## Preprocessing

- Trailing empty column (`Unnamed: 15`) dropped
- `user_id` dropped (identifier, not predictive)
- String columns whitespace-stripped
- Physically impossible values capped (e.g. `screen_time_hours > 24` → 24)
- Rows where `|screen_time_hours − work_screen_hours − leisure_screen_hours| > 1` flagged and logged but retained
- One-hot encoding for categorical features (`drop='first'`)
- `StandardScaler` for numeric features — fit on training set only

---

## Model Training

Four candidate models were trained and compared on identical 80/20 train/test splits:

1. **Linear Regression** (baseline)
2. **Ridge Regression**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**

Selection criterion: highest test-set R² (ties broken by lowest RMSE).  
Winning model hyperparameters were tuned with `RandomizedSearchCV` (n_iter=20, cv=5).  
Final model trained on full training set and re-evaluated on held-out test set.

### Final Model Metrics

> Values below are populated automatically after running `train_model.py`.  
> See `model/metrics.json` for exact numbers from your run.

| Metric | Value |
|---|---|
| **Best model** | *(see metrics.json → `best_model`)* |
| **Test R²** | *(see metrics.json → `tuned_test_r2`)* |
| **Test MAE** | *(see metrics.json → `tuned_test_mae`)* |
| **Test RMSE** | *(see metrics.json → `tuned_test_rmse`)* |
| **5-fold CV R²** | *(see metrics.json → best model's `cv_r2_mean`)* |

---

## Category Thresholds

Categories are derived from **training-set quartiles** of `mental_wellness_index_0_100`, not hardcoded:

| Category | Threshold |
|---|---|
| **Poor** | Score < Q25 (25th percentile of training data) |
| **Fair** | Q25 ≤ Score < Q50 |
| **Good** | Q50 ≤ Score < Q75 |
| **Excellent** | Score ≥ Q75 |

Exact values saved in `model/metrics.json → category_thresholds`.

---

## Known Limitations

1. **Small sample size (n = 400):** Estimates, especially for edge-case profiles, carry significant uncertainty. The model is not well-powered to detect subtle effects.

2. **Self-reported data:** All features and the target are self-reported. Social desirability bias, recall error, and anchoring effects are likely.

3. **Correlation ≠ causation:** Associations between screen time / sleep / stress and wellness scores do not establish causal relationships. Many confounders (income, health status, life events) are not captured.

4. **Not validated against clinical instruments:** The `mental_wellness_index_0_100` target has not been validated against clinical measures of depression, anxiety, or any DSM criteria.

5. **Right-skewed target:** Most participants score low (mean ≈ 20, median ≈ 15). Predictions for high-wellness users (score > 60) are extrapolated from sparse data and carry more uncertainty.

6. **No temporal modelling:** The dataset is cross-sectional. No causal or longitudinal inference should be drawn.

7. **Demographic generalisation:** The dataset's demographic distribution is unknown. Predictions may not generalise to all age groups, cultures, or occupational contexts.

---

## Ethical Considerations

- This model is **not a medical or diagnostic tool**. It must not be used to screen, assess, or label individuals for mental health conditions.
- Scores should be interpreted as **relative statistical estimates** within the distribution of the 400-participant survey, not as absolute measures of well-being.
- The disclaimer is shown on **every prediction result** and cannot be dismissed.
- No user data is stored, logged, or transmitted beyond the local API call.

---

## Explicit Statement

> **This model is NOT a medical or diagnostic tool.**  
> It is an educational demonstration of machine learning applied to self-reported lifestyle survey data.  
> Results should never be used as the basis for clinical decisions, mental health screening, or any consequential personal or institutional judgement.  
> Always consult a qualified mental health professional for personal advice.
