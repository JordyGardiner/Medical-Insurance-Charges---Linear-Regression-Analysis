## About This Repository

This is an archive of a completed academic project from my PDAN8411 module, uploaded here as part of my personal portfolio. The full Portfolio of Evidence (POE) is split into three parts; this repository currently holds Part 1, which covers the initial suitability check, analysis planning, EDA, and baseline/regularised regression modelling. Parts 2 and 3 will be added as separate notebooks/folders in due course.

## Project Summary

The project investigates whether linear regression is a suitable technique for predicting annual medical insurance charges (USD) from demographic and health-related features, using the Kaggle Medical Cost Personal Dataset (insurance.csv, 1,338 rows, 7 columns).

Rather than jumping straight into modelling, the notebook first tests the data against the assumptions of linear regression, builds a structured analysis plan, and only then proceeds through EDA, feature engineering, model training, and evaluation — each step backed by statistical evidence and academic referencing

## Dataset

| Feature  | Type                   | Notes                                      |
| -------- | ---------------------- | ------------------------------------------ |
| age      | Numerical (continuous) | 18–64 years                                |
| sex      | Categorical (binary)   | male / female                              |
| bmi      | Numerical (continuous) | 15.96–53.13                                |
| children | Numerical (discrete)   | 0–5                                        |
| smoker   | Categorical (binary)   | 20.5% yes, strong cost driver              |
| region   | Categorical (4 levels) | northeast, northwest, southeast, southwest |
| charges  | Numerical (target)     | Right-skewed (skewness ≈ 1.51)             |

One duplicate row was identified and removed, leaving 1,337 clean records with no missing values.

## What's in the Notebook

Task 1 — Suitability check. Tests linearity (Pearson correlation, raw vs. log-transformed target), skewness, outliers, and multicollinearity risk before committing to linear regression. Concludes regression is appropriate provided a log-transform of charges and a bmi × smoker interaction term are used.

Task 2 — Analysis plan. A six-phase roadmap (EDA → preprocessing → feature selection → model training → evaluation → report structure) that every later step follows.

Task 3 — Exploratory Data Analysis. Univariate/bivariate visualisations, correlation heatmap, and a formal feature-selection stage using four complementary methods:

Pearson correlation (numerical features)

Point-biserial correlation (binary features: sex, smoker)

One-way ANOVA F-test (region)

Variance Inflation Factor (multicollinearity, flagging VIF > 10)

Task 4 — Model training and evaluation. Trains OLS (baseline), Ridge (L2, tuned via RidgeCV), and Lasso (L1, tuned via LassoCV) on log-transformed charges with standardised features and an 80/20 train-test split (random_state=42). Evaluates with R², MAE, MSE, and RMSE (back-transformed to USD), plus residual diagnostics (residuals vs. fitted, Q-Q plot, residual histogram, actual vs. predicted). A second iteration adds an age² polynomial term and re-evaluates all three models.

## Key Results

| Model            | R²    | MAE (USD) | RMSE (USD) |
| ---------------- | ----- | --------- | ---------- |
| OLS (baseline)   | 0.837 | 4,187     | 8,699      |
| Ridge (L2)       | 0.838 | 4,107     | 8,387      |
| Lasso (L1)       | 0.837 | 4,161     | 8,633      |
| OLS v2 (+age²)   | 0.840 | 4,242     | 8,678      |
| Ridge v2 (+age²) | 0.840 | 4,218     | 8,610      |
| Lasso v2 (+age²) | 0.839 | 4,191     | 8,623      |

Ridge Regression (original feature set) was selected as the final model. Adding the age² term produced a marginal R² gain but increased MAE across all retrained models, indicating the original feature set (log target + bmi × smoker interaction) was already well specified.

Dominant predictors: the bmi × smoker interaction and age. sex and region contributed little signal (confirmed by near-zero point-biserial correlation and a non-significant ANOVA result, p = 0.251) and were retained deliberately so Lasso could objectively shrink them rather than dropping them by hand.

## Tech Stack
pandas · numpy · matplotlib · seaborn · scipy · statsmodels · scikit-learn



