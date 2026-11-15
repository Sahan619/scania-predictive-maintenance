# Preprocessing and Class Imbalance Handover

## Purpose

Prepare the Scania APS data for model training using reusable
preprocessing that can be fitted separately inside each
cross-validation training fold.

## Data and split

- Load CSV files using `skiprows=20` and `na_values="na"`.
- Encode non-APS (`neg`) as 0 and APS failure (`pos`) as 1.
- Non-APS does not necessarily mean a healthy vehicle.
- Split the original 60,000 training records using an 80/20
  stratified split with `random_state=42`.

| Partition | Records | Non-APS | APS failure |
|---|---:|---:|---:|
| Development | 48,000 | 47,200 | 800 |
| Internal holdout | 12,000 | 11,800 | 200 |

The official test CSV is reserved for the team's final evaluation.
The internal holdout has been used for preprocessing checks,
but not for the pilot model comparisons reported below.

Member 1's earlier EDA examined the full original training set,
so the internal holdout was not completely isolated from that EDA.

## Missing-value decisions

Retain all development rows for the initial approach.

There are 224 non-APS records and 2 APS failure records with more
than 100 missing readings. This threshold was used for inspection,
not as a deletion rule.

Retain highly missing columns initially because missingness differs
between classes. For example, `br_000` is missing in 83.40% of
non-APS records but only 9.25% of APS failure records.

Fill missing numeric values using training-set column medians.
Compare optional missing-value indicators experimentally.

No completely empty columns were found in the development data.

## Constant columns and scaling

`cd_000` has one observed value and 1.14% missingness.
Its numeric column becomes constant after median imputation and
is removed by `VarianceThreshold(threshold=0.0)`.

The missingness flag for `cd_000` is retained when indicators are
enabled because that flag varies across records.

Verified output sizes when fitted on the full development set:

- Without missing indicators: 169 features.
- With missing indicators: 338 features.

Feature counts can differ when fitted on other training subsets.

Scaling is optional. When enabled, StandardScaler scales all
retained features, including missing indicators.

The training-pipeline helper automatically enables scaling for
SMOTE because SMOTE uses distances.

## Imbalance experiments

The development class ratio is 59 non-APS records per APS failure.

Five-fold stratified cross-validation was run on development data
with shuffling and `random_state=42`.

All experiments used Logistic Regression with `solver="lbfgs"`
and `max_iter=2000`. Classification metrics and cost used the
model's default classification threshold.

Preprocessing was fitted separately within each training fold.
SMOTE was applied only to training folds.

SMOTE used `sampling_strategy=0.10` and `random_state=42`.
The ratio means one minority example per ten majority examples
after resampling.

### Mean validation results

| Approach | Precision | Recall | F1 | ROC-AUC | Average precision | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.8072 | 0.6587 | 0.7244 | 0.9564 | 0.7580 | 27554 |
| Class weighting | 0.3928 | 0.8863 | 0.5440 | 0.9538 | 0.7324 | 11302 |
| SMOTE | 0.6482 | 0.7700 | 0.7033 | 0.9479 | 0.7518 | 19072 |
| Baseline + indicators | 0.8168 | 0.6438 | 0.7195 | 0.9702 | 0.7665 | 28732 |
| Class weighting + indicators | 0.3694 | 0.9025 | 0.5241 | 0.9556 | 0.7354 | 10270 |

Cost = 10 × false positives + 500 × false negatives.

Reported cost is the mean cost per 9,600-record validation fold,
not the total across all five folds.

Average precision is reported explicitly. It is not the
trapezoidal area under the precision–recall curve.

Class weighting with indicators achieved the lowest mean cost
and highest recall among these pilot configurations, with lower
precision. The baseline achieved the highest F1.

These findings support further evaluation; they do not establish
the best configuration for every model. Fold-level scores are
available in the notebook's `fold_results` dictionary.

## Reusable functions

File: `src/preprocessing.py`

- `load_aps_csv(path)` returns numeric features and encoded labels.
- `split_development_data(X, y)` creates the reproducible split.
- `make_preprocessor(...)` builds preprocessing only.
- `make_training_pipeline(model, ...)` combines preprocessing,
  optional SMOTE, and a fresh copy of the supplied model.

### Example from the repository root

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.preprocessing import (
    load_aps_csv,
    split_development_data,
    make_training_pipeline,
)

X, y = load_aps_csv("data/aps_failure_training_set.csv")
X_dev, X_holdout, y_dev, y_holdout = split_development_data(X, y)

pipeline = make_training_pipeline(
    LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    ),
    scale=True,
    add_indicators=True,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(
    pipeline,
    X_dev,
    y_dev,
    cv=cv,
    scoring=["recall", "f1", "average_precision"],
    error_score="raise",
)
```

## Rules for model training

- Pass original, unprocessed features into cross-validation.
- Keep preprocessing inside the model pipeline.
- Never fit preprocessing or apply SMOTE before splitting into folds.
- Never resample validation, holdout, or official test records.
- Configure class weighting on the supplied model when supported.
- Use `class_weight="balanced"` to calculate weights during fitting
  for models that support this option.
- The helper deliberately rejects ordinary SMOTE combined with
  missing indicators, because interpolation can make binary flags
  fractional.
- The official test set must remain reserved until model selection
  and tuning are complete.

## Checks completed

- Verified split sizes, class counts, and disjoint row indices.
- Verified finite preprocessing outputs and preserved row counts.
- Verified removal of the constant numeric `cd_000` column.
- Verified optional scaling and missing indicators.
- Tested the training helper with weighted Logistic Regression
  plus indicators and with SMOTE.
- Verified fold-training medians and validation prediction counts.
- Verified the supplied estimator remains unfitted because the
  helper trains a clone.

### Model compatibility

The reusable preprocessing pipeline passed fit and prediction checks
with all six team models. Logistic Regression was checked earlier with
class weighting plus missing indicators and separately with SMOTE.

KNN, Gaussian Naive Bayes, Decision Tree, Random Forest, and XGBoost
each returned valid probabilities for all 9,600 rows in the same
development validation fold. These five checks used the following settings:

| Model | Scaling | Class weighting |
|---|---|---|
| KNN | On | None |
| Gaussian Naive Bayes | On | None |
| Decision Tree | Off | `class_weight="balanced"` |
| Random Forest | Off | `class_weight="balanced"` |
| XGBoost | Off | `scale_pos_weight` calculated from the training fold |

Missing indicators and SMOTE were disabled for these five checks.
The checks verified that the models work with the pipeline. 