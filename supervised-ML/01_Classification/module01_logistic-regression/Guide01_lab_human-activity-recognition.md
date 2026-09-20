# Module 1 — Lab Guide A
## Logistic Regression & Error Metrics: Human Activity Recognition

> **What this is.** A *standalone* walkthrough of the lab in `Guide01_Supervised-ML_Logistic-Regression_Introduction.ipynb`. You can read it start to finish to understand the entire lab without running anything, or keep it open beside the notebook and run each cell as you go. Every code block is annotated so you understand *what it does and why*.
>
> **Prerequisites.** Read the concepts guide (`Guide00_concepts_logistic-regression-and-classification-metrics.md`) first — this lab assumes you already understand logistic regression, regularization, and the classification metrics. You will also need the data file `Human_Activity_Recognition_Using_Smartphones_Data.csv` in the shared `data/` folder one level up from this module (the notebook reads `../data/...`). You also need `scikit-learn` 1.8 or newer.
>
> **Estimated time.** 45–75 minutes. One cross-validation step can take several minutes to run — that is expected.

---

## The problem in one paragraph

Researchers gave 30 volunteers a smartphone and asked them to perform six everyday activities — **walking, walking upstairs, walking downstairs, sitting, standing, and laying**. The phone's **accelerometer** (measures acceleration) and **gyroscope** (measures rotation) recorded motion, which was processed into **561 numeric features** per record (averages, extremes, frequencies, and so on). Our job: **given the 561 sensor features, predict which of the six activities the person was doing.** This is a six-class classification problem, and we will solve it with logistic regression — while paying close attention to how we *measure* success.

## Lab roadmap

| Step | What we do | Why it matters |
|---|---|---|
| **1** | Load and inspect the data | Understand data quality and structure before modeling |
| **2** | Analyze feature correlations | 561 features contain redundancy; find it |
| **3** | Split into train / test sets | The test set must simulate unseen data |
| **4** | Fit three models (no penalty, L1, L2) | Compare how regularization changes behavior |
| **5** | Visualize coefficients | See regularization shrink or zero out weights |
| **6** | Generate predictions and probabilities | Separate hard labels from probability scores |
| **7** | Compute error metrics | Go beyond accuracy — precision, recall, F1, AUC |
| **8** | Plot confusion matrices | See *which* activities get confused |

---

## Step 0 — Setup and reproducibility

We import our libraries and fix a **random seed**. Many steps involve randomness (the train/test split, the solver's starting point). Fixing the seed guarantees that you and everyone else get *identical* results every run — essential for reproducible science and for debugging.

```python
import warnings
import random
import numpy as np                       # arrays and math
import pandas as pd                      # tables (DataFrames)
import seaborn as sns                    # statistical plots
import matplotlib.pyplot as plt          # base plotting
from sklearn.exceptions import ConvergenceWarning

SEED = 42                                # any fixed number works, as long as it's fixed
random.seed(SEED)                        # seed Python's randomness
np.random.seed(SEED)                     # seed NumPy's randomness

# Hide noisy version-deprecation messages, but KEEP convergence warnings,
# which tell us if a model failed to finish optimizing.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('default', category=ConvergenceWarning)
```

> **Tip:** printing your library versions (`pd.__version__`, etc.) at the top of a notebook is a small habit that saves hours later — if a result changes, you can see whether a library upgrade caused it.

---

## Step 1 — Load and inspect the data

Four quick questions to answer before any modeling: What data *types* are the columns? Is the data already *scaled*? Are the classes *balanced*? And how do we turn the text labels into numbers?

```python
data = pd.read_csv("../data/Human_Activity_Recognition_Using_Smartphones_Data.csv", sep=',')
```

**Data types.** With 562 columns, we don't inspect them one by one — we count types:

```python
print(data.dtypes.value_counts())
```

You will see **561 `float64` columns** (the sensor features) and **1 `object` column** (the `Activity` label, stored as text). Good: the features are already numeric, so no conversion is needed there.

**Is the data already scaled?** Logistic regression is sensitive to feature magnitudes — a feature ranging 0–10,000 would overpower one ranging 0–1. The dataset documentation says every feature was normalized to $[-1, 1]$. We verify it instead of trusting it:

```python
print(data.iloc[:, :-1].min().value_counts())   # min of every feature column
print(data.iloc[:, :-1].max().value_counts())   # max of every feature column
```

`data.iloc[:, :-1]` means "all rows, all columns *except the last*" (the last is the label). If every one of the 561 feature minimums is `-1` and every maximum is `1`, the data is uniformly scaled and we can skip a scaling step. *(In practice a tiny numerical tolerance is fine — you are checking the range is essentially $[-1,1]$, not that every value is exactly $\pm 1$.)*

**Class balance.** If one activity dominated, accuracy would be a misleading metric (recall the 99%-healthy disease example from the concepts guide):

```python
data.Activity.value_counts()
```

Here the six activities appear in **roughly equal proportions** — the dataset is **balanced**. That is good news: it means overall accuracy is a *meaningful* headline number for this problem (we will still compute precision/recall/F1 for a fuller picture).

**Encode the labels as integers.** `scikit-learn` models need numeric targets, not strings like `"WALKING"`. `LabelEncoder` maps each activity name to an integer 0–5:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Activity'] = le.fit_transform(data.Activity)   # "WALKING" -> 3, etc.
data['Activity'].sample(5)                            # peek at a few encoded values
```

`fit_transform` does two things at once: `fit` *learns* the string→integer mapping, and `transform` *applies* it. Later you can translate predictions back to names with `le.inverse_transform([0, 1, ...])`.

---

## Step 2 — Analyze feature correlations

With 561 features, many measure nearly the same thing (redundancy). **Correlation** ($r$, from $-1$ to $+1$) measures how strongly two features move together. Highly correlated features add little new information and can make coefficient estimates unstable, so it helps to find them.

A correlation matrix is **symmetric** — the correlation of X with Y equals Y with X — and its diagonal is always 1 (every feature correlates perfectly with itself). So we keep only the **upper triangle** (excluding the diagonal) to count each pair once:

```python
feature_cols = data.columns[:-1]              # every column except the label
corr_values = data[feature_cols].corr()       # the full 561x561 correlation matrix

# Keep only the upper triangle (k=1 excludes the diagonal); the rest becomes NaN
upper_triangle = corr_values.where(
    np.triu(np.ones(corr_values.shape), k=1).astype(bool)
)

# Reshape from a wide matrix into a tidy 3-column table: feature1, feature2, correlation
corr_values = (upper_triangle
               .stack()                        # collapse the grid into one long column
               .to_frame()
               .reset_index()
               .rename(columns={'level_0': 'feature1',
                                'level_1': 'feature2',
                                0: 'correlation'}))

corr_values['abs_correlation'] = corr_values['correlation'].abs()  # magnitude only
```

We look at the strongest pairs and draw a histogram of correlation magnitudes:

```python
corr_values.sort_values('abs_correlation', ascending=False).head(10)   # top 10 pairs

plt.hist(corr_values.abs_correlation, bins=50)   # distribution of |r| across all pairs
plt.xlabel('Absolute Correlation'); plt.ylabel('Frequency')
plt.show()
```

**How to read the histogram.** A tall spike near 0 means most feature pairs are weakly related; a tail stretching toward 1.0 means some pairs are strongly collinear. Pairs with $|r| > 0.8$ are candidates for pruning:

```python
corr_values.sort_values('correlation', ascending=False).query('abs_correlation > 0.8')
```

> **Industry note.** Teams *review* highly correlated features but do not always delete them. For linear/logistic models, pruning improves coefficient stability and interpretability; for tree-based models, correlation is often tolerated. A common practice is to compare performance *with and without* pruning and keep the simpler feature set when scores are similar. In this lab we use correlation only as a diagnostic.

---

## Step 3 — Train / test split

We hold out **30%** of the data as a **test set** the model never sees during training — our stand-in for "new data in the real world." Because our classes are balanced and we want them to *stay* balanced in both halves, we use **`StratifiedShuffleSplit`**, which preserves each class's proportion across the split (a plain random split could accidentally skew them).

```python
from sklearn.model_selection import StratifiedShuffleSplit

strat_split = StratifiedShuffleSplit(n_splits=1,        # make one train/test split
                                     test_size=0.3,     # 30% goes to the test set
                                     random_state=42)   # reproducible split

# .split() yields index positions; next() pulls the first (and only) split
train_idx, test_idx = next(strat_split.split(data[feature_cols], data.Activity))

X_train = data.loc[train_idx, feature_cols]   # training features
y_train = data.loc[train_idx, 'Activity']     # training labels
X_test  = data.loc[test_idx,  feature_cols]   # test features
y_test  = data.loc[test_idx,  'Activity']     # test labels
```

**Convention:** capital `X` holds the input features; lowercase `y` holds the labels. Always verify the split preserved the balance:

```python
y_train.value_counts(normalize=True)   # proportions (sum to 1) in the training set
y_test.value_counts(normalize=True)    # proportions in the test set
```

The two tables should look almost identical — that is stratification doing its job.

---

## Step 4 — Fit three logistic regression models

To *see* what regularization does, we fit three variants and compare them:

| Model | Penalty | Behavior |
|---|---|---|
| `lr` | None | Coefficients unconstrained — risk of overfitting with 561 features |
| `lr_l1` | L1 (Lasso) | Drives weak coefficients to **exactly zero** (feature selection) |
| `lr_l2` | L2 (Ridge) | **Shrinks** all coefficients but keeps them non-zero |

Recall from the concepts guide that **`C = 1/λ`**: smaller `C` = stronger penalty. For the two regularized models we let **`LogisticRegressionCV`** search 10 values of `C` by 4-fold cross-validation and keep the best. All three use a **one-vs-rest (OvR)** setup: six binary classifiers, one per activity.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier

# One-vs-rest: OneVsRestClassifier trains one binary model per activity (six in total).

# 1) Unregularized baseline. C=np.inf means "no penalty" (scikit-learn 1.8 or newer).
lr = OneVsRestClassifier(
    LogisticRegression(C=np.inf, solver='lbfgs', max_iter=2000, random_state=SEED)
).fit(X_train, y_train)

# 2) L1-regularized (l1_ratios=[1] is pure L1), C chosen by cross-validation.
#    liblinear handles L1 for each of these binary sub-problems.
lr_l1 = OneVsRestClassifier(
    LogisticRegressionCV(Cs=10, cv=4, l1_ratios=[1], solver='liblinear',
                         scoring='accuracy', max_iter=2000, random_state=SEED)
).fit(X_train, y_train)

# 3) L2-regularized (l1_ratios=[0] is pure L2), C chosen by cross-validation.
lr_l2 = OneVsRestClassifier(
    LogisticRegressionCV(Cs=10, cv=4, l1_ratios=[0], solver='liblinear',
                         scoring='accuracy', max_iter=2000, random_state=SEED)
).fit(X_train, y_train)
```

> **Heads up — this is the slow part.** The L1 cross-validation can take **several minutes** on 561 features. That is normal and is itself a lesson: model cost grows with data and search size, which is why we tune deliberately rather than trying every option. L2 usually finishes faster than L1.

> **Version note.** This lab needs `scikit-learn` **1.8 or newer**. Two older arguments no longer work there: `multi_class` was removed in 1.8 (one-vs-rest is now written explicitly with `OneVsRestClassifier`), and `penalty` is deprecated in 1.8 (removal planned for 1.10), so the penalty is chosen with `l1_ratios` (or `l1_ratio` for a plain `LogisticRegression`), and `C=np.inf` means no penalty. Do **not** run this on scikit-learn 1.7 or older: there, `l1_ratio` is ignored unless `penalty='elasticnet'`, so an "L1" model would quietly be L2. The notebook checks your version for you; to check it yourself, run `import sklearn; print(sklearn.__version__)`, and upgrade with `pip install -U scikit-learn`.

---

## Step 5 — Visualize the coefficients

Each feature gets a coefficient (weight); its magnitude shows how strongly that feature affects the decision. Because this is a 6-class OvR problem, each model holds six fitted binary classifiers in `.estimators_`. Stacking their coefficient rows gives an array of shape **(6, 561)**, one 561-length row per activity. We assemble them into one tidy table, then plot the six class-specific coefficient sets:

```python
coefficients = []
coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab, mod in zip(coeff_labels, coeff_models):
    coeffs = np.vstack([est.coef_ for est in mod.estimators_])   # shape (6, 561): one row per OvR model
    # a two-level column label: which model, and which of the 6 class-vs-rest sets
    coeff_label = pd.MultiIndex(levels=[[lab], [0,1,2,3,4,5]],
                                codes=[[0,0,0,0,0,0], [0,1,2,3,4,5]])
    coefficients.append(pd.DataFrame(coeffs.T, columns=coeff_label))

coefficients = pd.concat(coefficients, axis=1)         # glue the three models side by side
```

```python
fig, axList = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
axList = axList.flatten()                               # 6 subplots, one per class

for loc, ax in enumerate(axList):
    # pull the 'loc'-th class-vs-rest coefficients for all three models
    coefficients.xs(loc, level=1, axis=1).plot(marker='o', ls='', ms=2.0,
                                               ax=ax, legend=(loc == 0))
    ax.set(title='Coefficient Set ' + str(loc))
plt.tight_layout()
```

**What to look for:**

- **`lr` (no penalty):** the widest spread of coefficient magnitudes.
- **`l1` (Lasso):** many coefficients sit *exactly on zero* — L1's built-in feature selection.
- **`l2` (Ridge):** coefficients are pulled toward zero but stay non-zero.

This is the whole point of the step: regularization visibly reshapes the model's parameters. In industry, these coefficient diagnostics support feature auditing, interpretability, and governance before deployment.

---

## Step 6 — Predictions and probabilities

We now score the held-out test set. We keep **both** hard labels (for accuracy, precision, recall, F1, confusion matrices) **and** full probability arrays (needed for ROC-AUC):

```python
y_pred  = []     # hard predicted labels, per model
y_prob  = []     # max class probability (a simple confidence proxy), per model
y_score = {}     # full (n_samples x 6) probability arrays, per model

for lab, mod in zip(coeff_labels, coeff_models):
    probs = mod.predict_proba(X_test)                  # shape (3090, 6): a prob for each class
    y_pred.append(pd.Series(mod.predict(X_test), name=lab))  # the argmax class
    y_prob.append(pd.Series(probs.max(axis=1), name=lab))    # confidence in the winning class
    y_score[lab] = probs                                     # keep the full array for AUC

y_pred = pd.concat(y_pred, axis=1)   # one column per model
y_prob = pd.concat(y_prob, axis=1)
y_pred.head()
```

The test set has 3090 rows, so `predict_proba` returns a 3090×6 array — for each record, the estimated probability of each of the six activities. The model's prediction is simply the class with the **highest** probability (`argmax`). Sanity-check the lengths line up:

```python
print(len(y_pred) == len(y_prob) == len(y_test))
```

---

## Step 7 — Compute the error metrics

For each model we compute accuracy, precision, recall, F1 (in both **weighted** and **macro** flavors), and a proper multi-class **ROC-AUC** from the probability scores. Because ROC-AUC needs probabilities (not labels), this is where the stored `y_score` arrays pay off.

```python
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

metrics = []
cm = {}

for lab in coeff_labels:
    # weighted = account for class frequency; macro = every class counts equally
    p_w, r_w, f_w, _ = score(y_test, y_pred[lab], average='weighted', zero_division=0)
    p_m, r_m, f_m, _ = score(y_test, y_pred[lab], average='macro',    zero_division=0)
    accuracy = accuracy_score(y_test, y_pred[lab])

    # multi-class AUC via one-vs-rest on the probability scores
    auc_w = roc_auc_score(y_test, y_score[lab], multi_class='ovr', average='weighted')

    cm[lab] = confusion_matrix(y_test, y_pred[lab])   # save for Step 8

    metrics.append(pd.Series({
        'accuracy': accuracy,
        'precision_weighted': p_w, 'recall_weighted': r_w, 'f1_weighted': f_w,
        'precision_macro': p_m,    'recall_macro': r_m,    'f1_macro': f_m,
        'auc_ovr_weighted': auc_w
    }, name=lab))

metrics = pd.concat(metrics, axis=1)   # models as columns, metrics as rows
metrics
```

**How to interpret the table.** Each column is a model; each row a metric. Ask:

- Are all three models similarly accurate, or does regularization clearly help or hurt?
- Are precision and recall balanced (F1 close to both), or is there a trade-off?
- Does the **L1** model stay competitive *despite* zeroing many coefficients? If so, the dropped features were genuinely redundant — a satisfying confirmation of the Step 2 correlation analysis.

Because the classes are balanced, expect high scores across the board (typically well above 0.9). The interesting differences are subtle and live in the confusion matrix.

---

## Step 8 — Confusion matrices: where the model fails

A single accuracy number hides *which* activities get confused. A **6×6 confusion matrix** shows it: rows are the true activity, columns the predicted one; the diagonal is correct, everything off-diagonal is an error. We **row-normalize** (divide each row by its total) so the diagonal reads directly as **per-class recall**.

```python
fig, axList = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
axList = axList.flatten()
axList[-1].axis('off')                        # we have 3 models, so hide the 4th panel

class_labels = list(le.classes_)              # translate 0-5 back to activity names

for ax, lab in zip(axList[:-1], coeff_labels):
    cm_norm = cm[lab].astype(float) / cm[lab].sum(axis=1, keepdims=True)  # row-normalize
    sns.heatmap(cm_norm, ax=ax, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    ax.set_title(f'{lab} (row-normalized)')
    ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
plt.tight_layout()
```

**The key finding.** Across all three models, the largest off-diagonal values are the confusion between **Sitting** and **Standing**. This makes physical sense: in both postures the phone is nearly still and similarly oriented, so the accelerometer and gyroscope produce **almost identical signatures**. A linear model simply cannot separate signals that overlap this much.

> **The diagnostic lesson.** When the *same* pair of classes is confused by *every* model (unregularized, L1, and L2 alike), the bottleneck is **feature separability**, not the choice of regularization. No amount of tuning fixes it — you would need *better features* (or a fundamentally different sensor) to tell sitting from standing. (If you own a smartwatch that occasionally nags you to "stand up" when you already are, you have met this exact limitation in the wild.)

---

## What you accomplished

You built a complete six-class classification pipeline: loaded and validated the data, hunted redundancy through correlation analysis, made a stratified split, trained and compared three regularization strategies, read the coefficients, and — most importantly — evaluated the models with the *right* tools, ending at a confusion matrix that revealed a real, physically-grounded limitation. That final move, from a single accuracy score to a diagnostic understanding of *where and why* a model fails, is the habit that separates a practitioner from a button-pusher.

**Next:** the second lab guide (`Guide02_lab_clinical-nutrition-multinomial.md`) applies the same skills to a **clinical nutrition** dataset with **imbalanced** classes — where accuracy becomes actively misleading and per-class metrics become essential.

---

*Lab guide prepared to accompany `Guide01_Supervised-ML_Logistic-Regression_Introduction.ipynb`. Notebook structure originally by Machine Learning Foundation © IBM Corporation; content written and modernized by Souvik Mandal, 2026.*
