# Regression: Linear Regression Notebooks

Six notebooks, one running example -- predicting Ames, Iowa house sale prices end to end -- following the [13-stage workflow](Guide00_Supervised-ML_Linear_Regression_end-to-end_workflow.md) from problem definition through deployment. Work through them in order; each one recaps and builds on the last.

## Reading Order

1. **Background (optional refresher):** [../Guide00_Introduction-to-Supervised-Machine-Learning.md](../Guide00_Introduction-to-Supervised-Machine-Learning.md) for the core concepts.
2. **Read the workflow guide:** [Guide00_Supervised-ML_Linear_Regression_end-to-end_workflow.md](Guide00_Supervised-ML_Linear_Regression_end-to-end_workflow.md). It walks through all 13 stages, from defining the problem to monitoring a deployed model, and shows which notebook practices each stage.
3. **Work through the notebooks** `Guide01` to `Guide06` in order (table below). Run them from this directory -- they import `pipeline/ames_workflow.py` and read from `data/` using relative paths.

## Notebooks

| Notebook | Stage(s) | Covers |
| :--- | :--- | :--- |
| `Guide01` | 1-5 | Define the problem, understand and clean the Ames data, split (2006-2009 train / 2010 test), explore the training set |
| `Guide02` | 6 | Impute, encode (ordinal + one-hot with rare-category grouping), engineer features, transform the target, scale |
| `Guide03` | 7-8 | Two baselines (mean, linear regression), cross-validated evaluation, residual diagnostics |
| `Guide04` | 9 (part 1) | Polynomial/interaction terms, the bias-variance trade-off, and what unchecked flexibility costs |
| `Guide05` | 9 (part 2) | Cross-validation pitfalls, Ridge and Lasso regularization, `GridSearchCV`, a frozen configuration |
| `Guide06` | 10-11 (+ 12-13) | The one-time final test, a model card, saving/reloading the pipeline, input validation |

Each notebook ends with a short **Your Turn** exercise applying that stage's idea to a second dataset (`CarPrice_Assignment.csv`, `encoded_car_data.csv`, or `california_housing_price.csv`).

## Directory Contents

| Path | What it holds |
| :--- | :--- |
| `pipeline/ames_workflow.py` | Shared code the notebooks import and build on across the series (`load_ames`, `clean_ames`, `split_ames`, `add_engineered_features`, `build_preprocessor`). Requires scikit-learn 1.1+. |
| `data/` | `Ames_Housing_Sales.csv` (the running example) plus three practice datasets used only in the Your Turn exercises. |
| `archive_v1/` | The previous, superseded set of five notebooks (each on its own dataset) and the retired Boston Housing data. Kept for reference, not part of the current reading order. |

## Setup

To run the notebooks locally, set up a Python virtual environment first. See [ML_PYTHON_JUPYTER_VENV_SETUP_GUIDE.md](../../ML_PYTHON_JUPYTER_VENV_SETUP_GUIDE.md).
