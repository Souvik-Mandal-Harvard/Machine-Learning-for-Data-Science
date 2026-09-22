# Supervised Machine Learning

This directory holds the reading material and Jupyter notebooks for supervised learning: regression first, then classification. Follow the order below.

## Reading Order

1. **Read the concept guide:** [Guide00_Introduction-to-Supervised-Machine-Learning.md](Guide00_Introduction-to-Supervised-Machine-Learning.md). It covers the ideas behind every notebook, so read it before writing any code.
2. **Work through `00_Regression/`.** Start with its [workflow guide](00_Regression/Guide00_Supervised-ML_Linear_Regression_end-to-end_workflow.md), then work through the `Guide01`-`Guide06` notebooks in order. The set follows one running example (Ames Housing) end to end; `Guide01` is ready, `Guide02` onward are being added. See the [regression README](00_Regression/README.md) for the notebook list.
3. **Then work through `01_Classification/`**, again in order of the `Guide` number.

Guide numbers restart inside each folder, so `00_Regression/Guide01` and `01_Classification/Guide01` are different notebooks. The notebooks are self-explanatory: read each task prompt before running the code cell under it.

## Directory Contents

| Path | What it contains |
| :--- | :--- |
| `Guide00_Introduction-to-Supervised-Machine-Learning.md` | Conceptual foundation: supervised learning, loss, generalization, regression metrics, workflow. |
| `00_Regression/` | Linear regression: a 13-stage workflow guide, notebooks `Guide01` onward (one running example, Ames Housing), shared helper code in `pipeline/`, the `data/` folder, and retired notebooks in `archive_v1/`. |
| `01_Classification/` | Logistic regression notebooks: `Guide01` (foundation) and `Guide02` (clinical case study). |

## Setup

To run the notebooks locally, set up a Python virtual environment first. See [ML_PYTHON_JUPYTER_VENV_SETUP_GUIDE.md](../ML_PYTHON_JUPYTER_VENV_SETUP_GUIDE.md).
