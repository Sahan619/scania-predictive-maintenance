# EDA Findings and Preprocessing Handover

## Project Information
- **Project:** Predictive Maintenance of Scania Truck Air Pressure Systems Using Supervised Machine Learning
- **Module:** EC5203 – Machine Learning
- **Dataset:** UCI APS Failure at Scania Trucks
- **EDA Responsibility:** Member 1 – Data Understanding and Exploratory Data Analysis

## 1. Dataset Overview

- Training records: 60,000
- Total columns: 171
- Sensor features: 170
- Target column: `class`
- Target classes: `neg` and `pos`

The dataset contains sensor measurements collected from Scania trucks to predict failures in the Air Pressure System (APS).

## 2. Class Distribution

| Class | Records | Percentage |
|---|---:|---:|
| Negative (`neg`) | 59,000 | 98.33% |
| Positive (`pos`) | 1,000 | 1.67% |

**Finding:** The dataset has severe class imbalance. Positive failure cases represent only 1.67% of the training data.

## 3. Missing-Value Analysis

- 169 out of 170 sensor features contain missing values.
- 59,409 records (99.015%) have at least one missing value.
- Only 591 records have complete sensor data.
- Maximum missing values in one record: 168.

### Features with the highest missing percentages

| Feature | Missing Percentage |
|---|---:|
| br_000 | 82.11% |
| bq_000 | 81.20% |
| bp_000 | 79.57% |
| bo_000 | 77.22% |
| cr_000 | 77.22% |
| ab_000 | 77.22% |
| bn_000 | 73.35% |
| bm_000 | 65.92% |
| bl_000 | 45.46% |
| bk_000 | 38.39% |

### Missing values per record

| Missing Values | Records |
|---|---:|
| None | 591 |
| 1–10 | 42,492 |
| 11–50 | 13,995 |
| 51–100 | 2,633 |
| More than 100 | 289 |

### Missingness by target class

| Class | Average Missing Values | Median |
|---|---:|---:|
| Negative | 13.91 | 8 |
| Positive | 29.15 | 20.5 |

Positive records have approximately twice as many missing sensor values on average as negative records.

Some features also show substantial class-specific differences in missingness:

| Feature | Negative Missing % | Positive Missing % |
|---|---:|---:|
| br_000 | 83.35% | 9.00% |
| bq_000 | 82.43% | 8.70% |
| bp_000 | 80.77% | 8.30% |
| bo_000 | 78.40% | 7.80% |
| bn_000 | 74.47% | 7.40% |
| bm_000 | 66.91% | 7.10% |
| ak_000 | 6.56% | 53.20% |
| ca_000 | 6.48% | 53.00% |
| di_000 | 5.90% | 52.40% |
| eb_000 | 5.90% | 52.40% |

**Finding:** Missingness patterns differ considerably between the two classes and may contain useful predictive information.

## 4. Feature Statistics and Distribution

- Sensor features have widely varying numerical scales.
- Several features have highly skewed distributions.
- Some features contain many zero values.
- Features such as `cs_009`, `cf_000`, and `ag_000` exhibit particularly high positive skewness.

These observations should be considered when selecting preprocessing and scaling techniques.

## 5. Outlier Analysis

Potential outliers were investigated using the Interquartile Range (IQR) method.

Some features with high potential outlier counts include:

| Feature | Potential Outliers |
|---|---:|
| az_007 | 14,235 |
| dr_000 | 13,007 |
| dq_000 | 12,870 |
| ba_008 | 12,580 |
| cj_000 | 12,540 |

These are statistical outliers and are not necessarily incorrect measurements. They should not automatically be removed.

## 6. Constant and Low-Variation Features

- `cd_000` is a constant feature with only one unique value.
- `ch_000` has two observed values: 0.0 (45,129 records) and 2.0 (10 records), with the remaining records missing.

These features should be investigated during feature selection.

## 7. Feature Correlation

Several highly correlated feature pairs were identified.

Examples:

| Feature 1 | Feature 2 | Correlation |
|---|---|---:|
| bb_000 | bv_000 | 1.000 |
| ah_000 | bg_000 | 1.000 |
| aa_000 | bt_000 | 1.000 |
| bx_000 | cc_000 | 0.996 |
| al_000 | am_0 | 0.994 |
| an_000 | ao_000 | 0.993 |

Highly correlated features may contain redundant information. Consider this during feature selection.

## 8. Feature Comparison by Class

Selected features showed substantial differences in median values between positive and negative records.

| Feature | Negative Median | Positive Median |
|---|---:|---:|
| aa_000 | 30,416 | 584,994 |
| ag_004 | 3,424 | 3,731,539 |
| ag_006 | 879,046 | 5,677,782 |
| ah_000 | 984,519 | 17,783,922 |

These differences suggest that some sensor measurements may be associated with APS failures. They do not independently establish predictive performance or causation.

## 9. Data Quality

- Duplicate records found: 0.
- One constant feature identified: `cd_000`.
- Missing values are widespread across the dataset.
- Sensor features have different scales and distributions.

## 10. Recommendations for Member 2 – Preprocessing

Based on the EDA findings, the following points should be considered:

1. **Missing-value handling:** Investigate suitable imputation strategies instead of removing all incomplete records.
2. **Feature selection:** Investigate highly missing, constant, low-variation, and highly correlated features.
3. **Missingness indicators:** Consider creating binary indicators for missing values because missingness patterns differ between target classes.
4. **Outlier handling:** Investigate extreme values carefully. Avoid automatically deleting potential outliers.
5. **Feature scaling:** Consider scaling for Logistic Regression and KNN. Tree-based models generally do not require feature scaling.
6. **Class imbalance:** Consider class weights and/or SMOTE to address the minority positive class.
7. **Data leakage prevention:** Split the data before fitting preprocessing transformations. Fit imputers and scalers using training data only, and apply resampling only to training data.
8. **Validation:** Use stratified splitting to preserve class proportions.

These are recommendations from EDA, not final preprocessing decisions.

## 11. Recommended Evaluation Metrics

Because the dataset is severely imbalanced, accuracy alone is not sufficient.

The following metrics should be considered:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Recall and PR-AUC are particularly important when evaluating the detection of actual APS failures.

## 12. Conclusion

The exploratory analysis identified severe class imbalance, extensive missing values, highly skewed sensor distributions, potential outliers, constant features, and strong correlations between several sensor features.

An important observation is that missing-value patterns differ considerably between positive and negative records. This may provide useful information for predictive modelling.

These findings provide the foundation for the preprocessing and feature-selection work carried out by Member 2.

**EDA notebook:** `notebooks/01_data_analysis.ipynb`

**Dataset source:** UCI Machine Learning Repository – APS Failure at Scania Trucks.
https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks