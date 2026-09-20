# Regression: Linear Regression Notebooks

This directory contains the linear regression reading material and hands-on Jupyter notebooks, `Guide00` to `Guide05`. Work through them in order.

## Reading Order

1. **Background (optional refresher):** [../Guide00_Introduction-to-Supervised-Machine-Learning.md](../Guide00_Introduction-to-Supervised-Machine-Learning.md) for the core concepts.
2. **Read the workflow guide:** [Guide00_Supervised-ML_Linear_Regression_end-to-end_workflow.md](Guide00_Supervised-ML_Linear_Regression_end-to-end_workflow.md). It walks through the 13-stage regression workflow, from defining the problem to monitoring a deployed model, and shows which notebook practices each stage.
3. **Work through the notebooks** `Guide01` to `Guide05` in order (table below).

## Notebooks

| Notebook | Topic | Dataset |
| :--- | :--- | :--- |
| `Guide01` | Transforming the target (log, square root, Box-Cox) and a first end-to-end model | California Housing (built into scikit-learn) |
| `Guide02` | Real workflow: cleaning, EDA, assumption checks, pipelines | `CarPrice_Assignment.csv` |
| `Guide03` | Encoding, train-test split, and feature scaling | `Ames_Housing_Sales.csv` |
| `Guide04` | Polynomial regression, pipelines, and grid search | `encoded_car_data.csv` |
| `Guide05` | Cross-validation, regularization (Lasso, Ridge), `GridSearchCV` | `boston_housing_clean.pickle` |

## Data

The `data/` folder holds the datasets the notebooks load. The notebooks use relative paths such as `data/CarPrice_Assignment.csv`, so run each notebook from this directory.

## Setup

To run the notebooks locally, set up a Python virtual environment first. See [ML_PYTHON_JUPYTER_VENV_SETUP_GUIDE.md](../../ML_PYTHON_JUPYTER_VENV_SETUP_GUIDE.md).
