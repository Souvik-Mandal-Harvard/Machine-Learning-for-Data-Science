# Module 1 — Lab Guide B
## Multinomial Logistic Regression: A Clinical Nutrition Case Study

> **What this is.** A *standalone* walkthrough of the lab in `Guide02_Supervised-ML_Logistic-Regression_Case-Study.ipynb`. Read it on its own to understand the whole lab, or run alongside the notebook cell by cell. Every code block is annotated.
>
> **Prerequisites.** Read the concepts guide (`Guide00_concepts_logistic-regression-and-classification-metrics.md`) and ideally the first lab guide (`Guide01_lab_human-activity-recognition.md`, Human Activity Recognition) first. You will need the data file `food_items.csv` in the shared `data/` folder one level up from this module (the notebook reads `../data/food_items.csv`), and `scikit-learn` 1.8 or newer.
>
> **Estimated time.** 30–60 minutes. Everything here runs in seconds (the dataset is small), so you can experiment freely.

---

## The problem in one paragraph

We have a dataset of **food items** described by their **nutritional profile** — 17 numbers per food (Calories, Total Fat, Saturated Fat, Sugars, Protein, Fiber, Sodium, vitamins, and so on). Each food carries a dietary recommendation for people managing **diabetes**: eat it **More Often**, **Less Often**, or only **In Moderation**. Our task: **predict the recommendation class from the nutrition numbers.** This is a three-class problem, and — unlike the balanced activity data of Lab A — the classes here are **imbalanced**, which changes how we must evaluate the model. There is also a real *cost asymmetry*: telling a diabetic patient to eat a high-sugar food "More Often" is a genuinely harmful mistake, not a neutral error. Keep that in mind throughout.

## Why this lab matters conceptually

Multinomial logistic regression is more than a toy: its **softmax** output is exactly the final layer of a classification neural network, and it remains a first-line model for tabular data in industry because it is fast, interpretable, and statistically well understood. By the end you should be able to reason about *why* each design choice is made, not just *how* to type it.

---

## Step 1 — Environment setup

Import everything up front and fix a random seed for reproducibility.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns

rs = 123   # random seed: makes splits and solver runs reproducible
```

> Fixing `random_state=rs` everywhere means the train/test split and the optimizer's starting point are identical across runs — so your results match this guide exactly.

---

## Step 2 — Exploratory Data Analysis (EDA)

**Never train before you understand the data.** Skipping EDA is one of the most common causes of models that fail in production. We ask three structured questions: What are the column types and ranges? What does the target distribution look like? And what do the features mean?

**2.1 Load and inspect.**

```python
food_df = pd.read_csv("../data/food_items.csv")
food_df.dtypes          # every nutrient should be numeric (float/int); 'class' is text (object)
food_df.head(10)         # look at 10 real rows to get a feel for the values
```

Store the feature names once, for clean, self-documenting code later:

```python
feature_cols = list(food_df.iloc[:, :-1].columns)   # all columns except the last ('class')
```

**2.2 Descriptive statistics.** Look at the *range* of each nutrient:

```python
food_df.iloc[:, :-1].describe()   # count, mean, std, min, quartiles, max per feature
```

You will notice the scales are wildly different — Calories run into the hundreds while a micronutrient sits in single digits. That mismatch is a problem for logistic regression (large-scale features dominate the optimization and make coefficients incomparable), and it directly motivates the **scaling** step coming up.

**2.3 Target class distribution.** This is the crux of the lab:

```python
food_df.iloc[:, -1:].value_counts(normalize=True)   # proportion of each class

counts = food_df.iloc[:, -1:].value_counts().reset_index()
counts.columns = ['class', 'count']
sns.barplot(data=counts, x='count', y='class')      # visualize the imbalance
plt.show()
```

The three classes are **not** equally represented — most foods are `In Moderation` or `Less Often`, while **`More Often` is the rare class.** This matters enormously: a classifier can score high *accuracy* just by favoring the common classes while quietly failing on the rare one. And here the rare class is the *beneficial* one — failing to identify "eat more of this" foods means patients miss genuinely good dietary choices. **That is why, throughout this lab, we track per-class precision, recall, and F1 rather than trusting accuracy alone.**

> **Industry note.** Production systems fight imbalance with oversampling (e.g. `SMOTE`), class weighting (`class_weight='balanced'`), or threshold calibration — techniques covered in later modules.

**2.4 Why multinomial (softmax)?** With three mutually exclusive classes, we use *multinomial* logistic regression: one joint model that gives each class a score and normalizes all scores into probabilities that sum to 1 via softmax,

$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^\top \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^\top \mathbf{x}}}, \qquad K = 3.$$

The alternative, **one-vs-rest**, trains three independent binary models whose scores need not sum to 1. We choose **multinomial** because a food belongs to exactly one category and we want calibrated, comparable probabilities.

---

## Step 3 — Feature engineering and preprocessing

Two transformations prepare the data for the model.

**3.1 Separate features and target.**

```python
X_raw = food_df.iloc[:, :-1]   # all rows, all columns except the last  -> features
y_raw = food_df.iloc[:, -1:]   # all rows, only the last column         -> target
```

*Reading `.iloc[rows, columns]`:* a colon `:` means "all," and `:-1` means "all except the last."

**3.2 Scale the features with `MinMaxScaler`.** This squeezes every feature into $[0, 1]$ using

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

```python
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)   # learn each feature's min/max, then rescale to [0,1]
print(f"Feature value range after scaling: [{X.min():.4f}, {X.max():.4f}]")
```

Scaling does two things: it helps the solver **converge faster**, and — crucially — it makes coefficients **comparable**, so a large coefficient really does mean "this nutrient matters more," rather than "this nutrient just happens to be measured in bigger units."

> **Data-leakage caveat.** We call `fit_transform` on the *whole* matrix here for simplicity in EDA. In a strict production pipeline you `fit` the scaler on the **training set only**, then `transform` both train and test — otherwise information from the test set leaks into preprocessing.

**3.3 Encode the target with `LabelEncoder`.** `scikit-learn` wants the target as integers:

```python
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.values.ravel())   # 'In Moderation'->0, 'Less Often'->1, 'More Often'->2
np.unique(y, return_counts=True)                          # confirm mapping and see the imbalance
```

`.ravel()` flattens the single-column DataFrame into the flat 1-D array the encoder expects. (We do **not** one-hot encode the *target* for `scikit-learn` — it handles integer class codes internally.)

---

## Step 4 — Train and evaluate the L2 (Ridge) model

**4.1 Stratified split.** We keep 20% for testing and use `stratify=y` so each split preserves the class proportions — vital here, because without it the rare `More Often` class could land almost entirely in one side by chance:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=rs)
print(f"Train: {X_train.shape},  Test: {X_test.shape}")
```

**4.2–4.3 Define and train.** Our baseline is an **L2-regularized multinomial** model. L2 shrinks all coefficients smoothly toward zero without eliminating any — a stable default when you expect most features to contribute.

```python
l2_model = LogisticRegression(
    random_state=rs,
    l1_ratio=0,                   # 0 = pure L2 (Ridge): sum of squared weights added to the loss
    solver='lbfgs',               # efficient quasi-Newton optimizer; supports L2; multinomial (softmax over all 3 classes) is the default
    max_iter=1000                 # cap on optimization steps (raise if it warns about convergence)
)
l2_model.fit(X_train, y_train)
l2_preds = l2_model.predict(X_test)
```

> **Version note.** This lab needs `scikit-learn` **1.8 or newer**. The old `multi_class='multinomial'` argument was removed in 1.8: multinomial (softmax) is simply the default for the `lbfgs` and `saga` solvers, so we omit it. The `penalty` argument is deprecated in 1.8 (removal planned for 1.10); the penalty is chosen with `l1_ratio` instead (`0` = L2, `1` = L1, in between = Elastic-Net). Do **not** run this on scikit-learn 1.7 or older: there, `l1_ratio` is ignored unless `penalty='elasticnet'`, so an "L1" model would quietly be L2. Upgrade with `pip install -U scikit-learn`.

**4.4 A reusable evaluation helper.** We report **per-class** precision, recall, and F1 — not a single averaged number — because that is the only way to see how the model treats the rare class:

```python
def evaluate_metrics(yt, yp):
    """Return accuracy plus per-class precision, recall, and F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(yt, yp)  # arrays, one value per class
    return {'accuracy': accuracy_score(yt, yp),
            'precision': precision, 'recall': recall, 'f1score': f1}

pd.DataFrame(evaluate_metrics(y_test, l2_preds))
```

**Reading the L2 results.** Overall accuracy is decent (around **0.77**), but that headline hides the real story: **recall for class 2 (`More Often`) is noticeably lower** than for the other two classes. With fewer training examples of beneficial foods, the model has less evidence to learn that boundary and misses many of them — a *false negative* with real clinical consequences. This is exactly the danger the EDA warned about.

---

## Step 5 — The L1 (Lasso) model: feature selection via sparsity

L1 differs from L2 in one powerful way: it can push coefficients to **exactly zero**, effectively deleting uninformative features — **automatic feature selection**. In a nutrition dataset with correlated features (Total Fat, Saturated Fat, and Trans Fat all measure dietary fat), that can improve generalization by removing redundancy.

```python
l1_model = LogisticRegression(
    random_state=rs,
    l1_ratio=1,                   # 1 = pure L1 (Lasso): sum of absolute weights
    solver='saga',                # REQUIRED for L1 + multinomial (lbfgs does NOT support L1)
    max_iter=1000
)
l1_model.fit(X_train, y_train)
l1_preds = l1_model.predict(X_test)
```

> Note the **solver had to change** to `saga`. Penalty and solver are not independent choices — `lbfgs` cannot do L1. If you ever see a solver error, this mismatch is the usual cause.

**5.1 Inspecting probabilities.** Beyond the hard prediction, `predict_proba` reveals the model's *confidence*:

```python
l1_model.predict_proba(X_test[:1, :])[0]   # e.g. [0.02, 0.96, 0.02] -> very confident it's class 1
```

The three numbers are the softmax probabilities for classes 0, 1, 2; they sum to 1, and `predict()` simply returns the `argmax`. A confident 0.96 and a hesitant 0.52 are both "class 1," but in a clinical decision-support system a low-confidence prediction might be flagged for human review rather than acted on automatically.

**5.2 Evaluate and compare.**

```python
pd.DataFrame(evaluate_metrics(y_test, l1_preds))
```

The L1 model typically **outperforms** the L2 baseline (accuracy around **0.81**). The likely reason: L1 zeroed out coefficients for correlated or noisy nutrients, yielding a simpler model that generalizes better. This is the same instinct an analyst has when dropping redundant columns by hand — but L1 does it automatically and in a principled, data-driven way.

**L1 vs. L2 — a practical decision guide:**

| Situation | Prefer |
|---|---|
| Many features, only a few truly predictive | **L1** (feature selection, model compression) |
| All features expected to contribute | **L2** (stable shrinkage) |
| Highly correlated features | **L1** or Elastic-Net |
| Need a small, deployable model | **L1** (zeros reduce computation) |
| Coefficient stability matters most | **L2** (L1 solutions can jump with small data changes) |

The penalty strength itself (`C = 1/λ`) is a hyperparameter you would tune with cross-validation (`LogisticRegressionCV`) in a real project.

---

## Step 6 — Confusion matrix: where does the model fail?

Aggregate metrics say *how well*; the confusion matrix says *which* mistakes. For three classes it is a 3×3 grid — rows = true class, columns = predicted. We **row-normalize** so the diagonal reads as **per-class recall**:

```python
cf = confusion_matrix(y_test, l1_preds, normalize='true')   # each row sums to 1
disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=l1_model.classes_)
disp.plot(cmap='plasma')
plt.title("Normalized Confusion Matrix — L1 Logistic Regression")
plt.show()
```

Read it row by row: the diagonal cell of each row is that class's recall; large off-diagonal cells show *what it gets confused with*. Expect the `More Often` row to show the weakest diagonal — the imbalance again — with its errors leaking into the more common categories.

> **Industry use.** Confusion matrices are a standard review deliverable: teams use them to prioritize the most *costly* errors, justify threshold changes, monitor drift over time, and communicate with non-technical stakeholders (a clinician grasps a confusion matrix far faster than an F1 score).

---

## Step 7 — Interpreting the model through coefficients

Logistic regression's superpower over black-box models is **interpretability**. Each coefficient tells you how a one-unit increase in a (scaled) nutrient shifts the log-odds of a class:

$$\log\frac{P(y=k \mid \mathbf{x})}{P(y=0 \mid \mathbf{x})} = \mathbf{w}_k^\top \mathbf{x} + b_k$$

- **Large positive** coefficient → that nutrient pushes predictions *toward* class $k$.
- **Large negative** → pushes *away* from class $k$.
- **Zero** (from L1) → deemed uninformative for that class.

The coefficient array has shape `(3 classes, 17 features)`; many entries are exactly `0.0` thanks to L1:

```python
pd.DataFrame(l1_model.coef_).transpose()
```

Because the raw array is hard to read, the notebook provides helpers (`get_feature_coefs`, `get_bar_colors`, `visualize_coefs`) that extract the non-trivial coefficients for one class and draw a horizontal bar chart (green = pushes toward the class, red = pushes away):

```python
coef_dict = get_feature_coefs(l1_model, 1, feature_cols)   # class 1 = 'Less Often'
visualize_coefs(coef_dict, title="Feature Coefficients for Class 1: 'Less Often'")
```

**Do the results make clinical sense?** For **`Less Often`**, nutrients like **Saturated Fat, Sugars, Cholesterol, and Total Fat** carry large *positive* coefficients — foods high in these are pushed toward "eat less often," exactly matching dietary guidelines for diabetics. For **`More Often`** (class 2), high **Calories, Total Carbohydrate, and Total Fat** are strongly *negative* — energy-dense foods are unlikely to be "eat freely." When a model's learned coefficients agree with domain knowledge, that is a reassuring sign it learned real signal; when they *contradict* it, that is a red flag worth investigating (a data issue, a confounder, or a "Clever Hans" model that is right for the wrong reason).

> **Why interpretability is not just academic.** In healthcare, explainability is required under FDA guidance for AI-based medical software; in finance, regulations like GDPR Article 22 demand explanations for automated decisions. A model that is *good enough and interpretable* is often preferable to a marginally better black box.

---

## Step 8 — Your turn: Elastic-Net (coding exercise)

Now apply the same **Define → Train → Evaluate → Interpret** workflow to a third strategy: **Elastic-Net**, which blends L1 and L2:

$$\mathcal{L}_{\text{EN}} = \text{Cross-Entropy} + \lambda\Big[\alpha \sum |w_i| + (1-\alpha)\sum w_i^2\Big]$$

where `l1_ratio` = $\alpha$ sets the mix: `1.0` = pure L1, `0.0` = pure L2, and anything in between blends them. Elastic-Net shines when features form *correlated groups* — it tends to keep or drop whole groups together rather than arbitrarily picking one member (as pure L1 does).

```python
en_model = LogisticRegression(
    random_state=rs,
    solver='saga',        # only saga supports elastic-net
    max_iter=1000,
    l1_ratio=0.5          # 50/50 blend; try 0.1 and 0.9 too and compare
)
en_model.fit(X_train, y_train)
en_preds = en_model.predict(X_test)
pd.DataFrame(evaluate_metrics(y_test, en_preds))     # compare per-class F1 with L1 and L2
```

Then plot its confusion matrix and coefficients exactly as in Steps 6–7. **Questions to reflect on:** Does Elastic-Net match or beat pure L1 on this data? Does it zero out more or fewer features? How does per-class recall for the hard `More Often` class change as you vary `l1_ratio`? Elastic-Net usually leaves *more* non-zero coefficients than L1 but *fewer* than L2.

---

## What you accomplished

You ran a full multinomial classification pipeline on messy, real-world, **imbalanced** clinical data — and, more importantly, you evaluated it *honestly*. The recurring theme: **accuracy alone would have hidden the model's weakness on the rare, clinically important `More Often` class.** Per-class metrics, a normalized confusion matrix, and interpretable coefficients together told the true story — and connected the model's math back to real nutrition knowledge and real patient consequences.

**Looking ahead.** The rest of the course introduces classifiers with different assumptions — decision trees, SVMs, KNN, ensembles. For each, ask: What does it assume about the data? How does its decision boundary differ from logistic regression's straight line? And when is its added complexity actually worth it? Logistic regression's linear boundary is both its limit and its strength — fast, transparent, and the baseline every fancier model must beat.

---

*Lab guide prepared to accompany `Guide02_Supervised-ML_Logistic-Regression_Case-Study.ipynb`. Notebook structure originally by Yan Luo © IBM Corporation, 2021; content written and modernized by Souvik Mandal, 2026.*
