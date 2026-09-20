# Guide00: The End to End Linear Regression Workflow Concepts

This guide turns the ideas from the [Introduction to Supervised Machine Learning](../Guide00_Introduction-to-Supervised-Machine-Learning.md) into a concrete, repeatable recipe for a regression project, from the first question to a model that is running, monitored, and improved. Read it before opening the notebooks: it is the map, and each notebook (`Guide01`–`Guide05`) walks through one part of the territory in depth.

By the end of this guide, you should be able to:
* name the 13 stages of a typical regression project and say what happens in each,
* explain why the train-test split is its own stage and why it comes before exploration and transformation,
* explain what cross-validation, regularization, and grid search each add, and why they come after a baseline,
* explain why the workflow is a loop, not a straight line,
* find which notebook practices each stage.

---

## 1. The Workflow at a Glance

![The linear regression workflow in 13 stages: before modeling (define, understand, clean, split, explore, transform), modeling (baseline, evaluate, improve and select, final test and report), and operating (deploy, monitor and feedback, implement feedback), with two return loops.](linear-regression_workflow_stages.svg)

| # | Stage | What happens | Practiced in |
| :-- | :--- | :--- | :--- |
| 1 | Define the problem | Question, target, metric, objective | all notebooks (problem statements) |
| 2 | Understand the data | Source, meaning of each column and row | `Guide02`, `Guide03` |
| 3 | Clean the data | Fix errors that do not require learning from data | `Guide02` |
| 4 | Split | Separate training and test sets | `Guide01`, `Guide02`, `Guide03` |
| 5 | Explore | Distributions, relationships, early assumption checks | `Guide02` |
| 6 | Transform | Impute, encode, engineer, transform target, scale | `Guide01`, `Guide03`, `Guide04` |
| 7 | Baseline model | Fit a simple model to beat | `Guide01`, `Guide02`, `Guide04` |
| 8 | Evaluate | Metrics and residual diagnostics | `Guide01`, `Guide02` |
| 9 | Improve and select | Cross-validation, regularization, grid search | `Guide04`, `Guide05` |
| 10 | Final test and report | One look at the test set; communicate results | `Guide05` (partly) |
| 11 | Deploy | Put the whole pipeline into use | not covered |
| 12 | Monitor and collect feedback | Watch for drift; gather user and expert input | not covered |
| 13 | Implement feedback | Update the data or the model, then retest | not covered |

Not every project needs every sub-step of Stage 6. Apply only what your data calls for (missing values, categorical columns, curved relationships, a skewed target, features on very different scales). The notebooks also revisit the same stages with increasing depth rather than following this order strictly, and `Guide02` explores the data before it splits.

**A note on vocabulary.** Throughout this guide, *stage* means one of the 13 lifecycle stages above. Within a stage, the individual techniques (for example, one-hot encoding within Stage 6) are just called techniques.

---

## 2. Two Rules That Run Through Everything

### Rule 1: Learn from training data only
The test set stands in for future, unseen data. If any information from it leaks into training, your test score is no longer an honest estimate of real-world performance. This is called **data leakage**.

Anything that *learns something from data* must be fit on the training set only, then applied to the test set:
* the median an imputer computes,
* the categories an encoder discovers,
* the mean and standard deviation a scaler computes,
* the $\lambda$ found by a Box-Cox transformation.

In scikit-learn terms: call `fit` / `fit_transform` on training data, and only `transform` on test data. This is why the split (Stage 4) comes before exploration and transformation.

### Rule 2: Touch the test set once
Tuning decisions (which polynomial degree, which regularization strength) are made using the *training* data, via cross-validation. The test set is used a single time, at the end of Stage 10. If you keep adjusting the model until the test score looks good, the test set has become part of training.

---

## 3. Stages 1–3: Define, Understand, and Clean

### Stage 1. Define the Problem
State the question in plain language ("How much will this house sell for?"). Then pin down:
* the **target** ($y$, a continuous quantity) and the candidate **features** ($X$),
* the **success metric** (for example, MAE in dollars), chosen *before* you see any results,
* the **objective**: *interpretation* (why does $y$ change?) or *prediction* (how accurate is the forecast?), since that shapes how you model,
* what a **good enough** result looks like, and how a prediction will be used.

### Stage 2. Understand the Data
Before touching the data, learn where it came from and what it means:
* the **source** and how it was collected (surveys, sensors, transactions), including known biases and the time period covered,
* what **one row** represents (a house? a sale? a customer-month?),
* what each **column** means, its unit, and its type (numeric, categorical, date),
* how the **target** was defined and measured.

This context tells you what is plausible, which values are errors, and whether the model will generalize to where you want to use it (a model trained on Ames, Iowa says little about Boston).

### Stage 3. Clean the Data
Handle problems in the raw data: duplicate rows, inconsistent labels (for example, `"toyota"` vs. `"Toyota"`), wrong data types, impossible values, and unit mistakes. Investigate outliers before removing them; they may be errors, or they may be the most informative rows.

Only do fixes here that **do not learn from the data** (removing duplicates, correcting typos, fixing units). Fixes that *do* learn from it, such as filling missing values with a median, belong in Stage 6, after the split.

---

## 4. Stages 4–6: Split, Explore, and Transform

### Stage 4. Split Into Training and Test Sets
Split into `X_train`, `X_test`, `y_train`, `y_test` (for example, 80/20). Fix `random_state` so results are reproducible. From here on, **the test set is off limits** until Stage 10: no plots, no statistics, no tuning.

The split comes *before* exploration because what you see while exploring shapes the choices you make (which features to keep, which transformations to try). If you explore the test rows, they influence those choices.

### Stage 5. Explore the Data (Training Set Only)
Use summary statistics and plots on the training set to understand its structure:
* the **distribution of the target** (is it strongly skewed?),
* the **relationship between each numeric feature and the target** (roughly linear, or curved?),
* **correlations among features** (severe multicollinearity makes coefficients unstable),
* the **spread of the target** across categories or ranges (a first hint of non-constant variance).

This is where you decide which parts of Stage 6 your data needs.

### Stage 6. Transform the Data
Linear regression needs a numeric, complete, reasonably well-behaved feature matrix. This stage gets your data into that shape. **Every technique here is fit on `X_train` and only applied to `X_test`.** A sensible order is: impute → encode → engineer → scale (the target is transformed separately).

**6.1 Handle missing values.** Most scikit-learn models cannot handle missing values.
* *Impute:* fill with the median (numeric, robust to outliers), the mean, or the most frequent value (categorical). Compute the fill value from `X_train` only.
* *Drop* rows or columns, when only a few are affected or a column is mostly empty.
* *Add a "was missing" indicator* when the missingness itself carries information.

**6.2 Encode categorical features.** Linear regression needs numbers.
* *Nominal categories* (no natural order, such as car brand): one-hot encoding creates one 0/1 column per category. For plain linear regression, drop one level per category so coefficients stay uniquely interpretable. Set the encoder to ignore categories that appear only in the test data.
* *Ordinal categories* (natural order, such as "poor < fair < good"): map to ordered integers.
* *Many rare categories:* group rare levels into an "other" bucket first. One-hot encoding adds one column per level, which can inflate the feature count and cause overfitting.

**6.3 Engineer features (polynomials and interactions).** To capture non-linear relationships while keeping a linear model, create new features:
* *Polynomial terms* ($x^2$, $x^3$) let the fitted line curve.
* *Interaction terms* ($x_1 \cdot x_2$) let one feature's effect depend on another.
* *Domain-driven features* (ratios, differences, "age of house" from a year column) often help more than any automatic method.

`PolynomialFeatures` learns nothing from the data, so it cannot leak by itself, but following the fit-on-train habit is the safer default. Watch the column count: it grows quickly with degree and with the number of features, which raises the risk of overfitting. Stage 9 addresses that risk.

**6.4 Transform the target.** Linear regression does not require the *target* to be normal. Normally distributed *residuals* matter mainly for inference (confidence intervals and p-values). Still, transforming a strongly skewed target (typical for prices and incomes) often stabilizes the error variance and improves the fit.
* Check the distribution of `y_train` (histogram, `stats.normaltest`).
* If it is skewed, apply a log, square-root, or Box-Cox transformation to `y_train` only.
* **Crucial:** save the transformation parameters (like the Box-Cox $\lambda$) so you can invert it when you predict.

**6.5 Scale the features.** Standardize (zero mean, unit variance) or normalize (fixed range) the features.
* Plain least-squares predictions do not change with feature scale, but scaling **matters for regularization** (Stage 9: the penalty treats all coefficients equally, so features must be comparable), for gradient-based solvers, and for comparing coefficient sizes.
* Fit the scaler (e.g., `StandardScaler`) on `X_train`; use it to transform both `X_train` and `X_test`.

---

## 5. Stages 7–8: Baseline Model and Evaluation

### Stage 7. Baseline Model
Always start with a simple model. It gives you a benchmark that every later improvement must beat.
* **Fit:** train the linear regression on the processed training data (the transformed `X_train` and the transformed `y_train`). This is where the algorithm minimizes the loss and estimates the parameters ($\beta_0, \beta_1, \dots, \beta_p$).
* **Predict:** pass the processed `X_test`, transformed with the *training-fitted* tools from Stage 6, to the model to get predictions $\hat{y}$.
* **Convert back:** if you transformed the target, the predictions are still on the transformed scale (for example, log-dollars). Use the saved parameters to convert them to the original units (dollars).

### Stage 8. Evaluate
Compare the **converted** predictions against the **original, untouched** `y_test`, using several metrics: MAE, RMSE, and $R^2$. Metrics computed on the transformed scale are not comparable to metrics from a model without a transformation. Also compare training and test error: a much lower training error signals overfitting.

Numbers tell you *how much* error there is; residual plots tell you *what kind*. Analyze the residuals ($y - \hat{y}$):
* **Linearity and constant variance (homoscedasticity):** residuals vs. fitted values should show no curve and no funnel.
* **Normality of residuals:** a Q-Q plot (matters mainly for inference).
* **Outliers and influential points:** a few extreme residuals can distort the fit.

A curve suggests missing non-linear terms; a funnel suggests a target transformation. Both send you back to Stage 6 (Loop 1, see Section 9).

---

## 6. Stage 9: Improve and Select

A single train-test split gives one estimate, and it depends on which rows landed where. This stage makes model comparison and tuning reliable, using the *training* data only.

### 9.1 Cross-Validation
In **k-fold cross-validation**, split the *training* set into $k$ folds. Train on $k-1$ folds, validate on the remaining fold, and rotate until every fold has been the validation set once. Averaging the $k$ scores gives a more stable estimate, and their spread tells you how sensitive the model is to the data it sees.

* The test set stays untouched.
* Preprocessing must be **re-fit inside each fold** (otherwise the scaler or encoder leaks validation data). Doing this by hand is tedious and error-prone, which is why we use a `Pipeline` (see Section 10).

### 9.2 Regularization (Ridge and Lasso)
As you add features (Stage 6.3), the model can start fitting noise: **overfitting**. Regularization discourages this by adding a penalty on large coefficients to the loss:

$$\text{loss} = \text{prediction error} + \alpha \times \text{penalty}(\beta)$$

| Method | Penalty | Effect |
| :--- | :--- | :--- |
| **Ridge** (L2) | sum of squared coefficients | Shrinks all coefficients toward zero; handles correlated features well. |
| **Lasso** (L1) | sum of absolute coefficients | Can shrink some coefficients exactly to zero, acting as automatic feature selection. |

(scikit-learn scales the terms slightly differently between the two, but the idea is the same.) The **regularization strength $\alpha$** is a *hyperparameter*: $\alpha = 0$ is ordinary linear regression, and larger $\alpha$ means a simpler, more constrained model (more bias, less variance). Features must be scaled first (Stage 6.5), or the penalty would unfairly punish features measured in small units.

### 9.3 Hyperparameter Tuning with Grid Search
You now have several hyperparameters to choose: the polynomial degree, $\alpha$, which method (Ridge or Lasso), and so on. **Grid search** automates the choice:
1. List candidate values for each hyperparameter (the "grid"). For $\alpha$, use a log-spaced range such as 0.01, 0.1, 1, 10, 100.
2. For every combination, run cross-validation on the training set.
3. Pick the combination with the best average validation score, then refit on the full training set.

In scikit-learn this is `GridSearchCV`. Its cost is (number of combinations) × $k$ model fits, so keep grids sensible. Note that the best cross-validation score is slightly optimistic, because you picked it as the best of many; comparing tuned models fairly calls for *nested* cross-validation.

---

## 7. Stage 10: Final Test and Report

### Final test
Take the single best tuned model, predict on the test set **once**, convert predictions back to original units, and report MAE, RMSE, and $R^2$. Check the residuals again. This number is your honest estimate of performance on new data. Before deployment, you may retrain the chosen configuration on all available data.

### Report
A result nobody understands does not get used. Communicate in the language of the people who will act on it:
* the **error in original units** ("typically off by about \$18,000"), not just $R^2$,
* the **most influential features** and the direction of their effects (for an interpretation objective, coefficients in original units with the "holding other features constant" caveat),
* the **limitations**: where the model is weak, which assumptions were violated, and what data it was trained on,
* enough **documentation to reproduce** it: data version, code, `random_state`, and the chosen hyperparameters.

---

## 8. Stages 11–13: Deploy, Monitor, and Implement Feedback

These stages happen after the notebooks end, but they explain why the earlier stages are done the way they are.

### Stage 11. Deploy
Put the model where it can be used: a batch job that scores a file each night, an API, a dashboard.
* **Save the whole fitted pipeline** (preprocessing, target transformation, and model), not just the model. New data must go through exactly the same steps as the training data.
* Validate incoming data (expected columns, types, ranges) before it reaches the model.
* Version the model, the data it was trained on, and the code that built it.

### Stage 12. Monitor and Collect Feedback
A deployed model degrades silently, because the world changes.
* **Monitor inputs (data drift):** are feature distributions moving away from the training data? Are new categories appearing? Are missing values increasing?
* **Monitor performance:** when true outcomes become known (the house actually sells), compare them with the predictions and track MAE over time.
* **Collect feedback:** user complaints, domain-expert review, and error analysis by segment (does the model fail badly on one neighborhood or price range?).

### Stage 13. Implement Feedback
Turn what you learned into a change, then send it back through the pipeline. Decide where the problem lives:

| The feedback shows... | Go back to |
| :--- | :--- |
| the question or metric no longer matches the need | Stage 1 |
| a data problem (new sources, errors, missing coverage) | Stage 2–3 |
| a feature or transformation problem | Stage 6 |
| a model or tuning problem | Stage 9 |

Whatever you change, re-run the evaluation (Stage 8) and the final test (Stage 10) before redeploying, and record the new version.

---

## 9. Iteration: The Two Loops

The workflow is a loop, not a straight line. Two loops matter most (they appear as arrows in the diagram):

* **Loop 1 (inner): diagnostics send you back to Stage 6.** A curved residual plot means you need new features; a funnel means the target needs a transformation. This loop runs many times during development, always using the training data and cross-validation.
* **Loop 2 (outer): monitoring and feedback send you back to the data.** After deployment, drift and user feedback reveal what the training data missed. This loop can restart the project at Stage 1, 2, or 3, and every pass ends with a fresh final test.

---

## 10. Putting It Together: Two Code Sketches

### Sketch A: Stages 4–8 by hand
A minimal illustration of the fit-on-train discipline. It assumes `X` is already numeric and complete, and `y > 0` (a Box-Cox requirement):

```python
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Stage 4: split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Stage 6.3: feature engineering (fit on train, apply to both)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_p = poly.fit_transform(X_train)
X_test_p = poly.transform(X_test)

# Stage 6.5: scaling (fit on train only)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_p)
X_test_s = scaler.transform(X_test_p)

# Stage 6.4: target transformation (lambda estimated from train only)
y_train_bc, lam = stats.boxcox(y_train)

# Stage 7: fit, predict, and convert back to original units
model = LinearRegression().fit(X_train_s, y_train_bc)
y_pred = inv_boxcox(model.predict(X_test_s), lam)

# Stage 8: evaluate against the untouched y_test
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2 :", r2_score(y_test, y_pred))
```

Writing every step by hand is the best way to learn what each one does, but it is error-prone, and it does not scale to cross-validation: you would have to repeat all of it inside every fold.

### Sketch B: Stages 4–11 as one pipeline
A scikit-learn **`Pipeline`** (with a `ColumnTransformer` for mixed column types) chains the transformations and fits them on training data only, and, inside `GridSearchCV`, re-fits them within every cross-validation fold. `TransformedTargetRegressor` applies the target transformation and its inverse automatically. This sketch assumes `X` is a DataFrame, with `num_cols` and `cat_cols` listing its numeric and categorical columns, and `y > 0`:

```python
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

# Stage 4: split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Stage 6: transformations, one branch per column type
numeric = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scale", StandardScaler()),
])
categorical = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])

# Stages 6.4, 7, 9.2: target transform + preprocessing + regularized model
model = TransformedTargetRegressor(
    regressor=Pipeline([("prep", preprocess), ("reg", Ridge())]),
    func=np.log1p, inverse_func=np.expm1,          # log-transform the target, invert on predict
)

# Stages 9.1, 9.3: cross-validated grid search over hyperparameters (training data only)
grid = GridSearchCV(
    model,
    param_grid={
        "regressor__prep__num__poly__degree": [1, 2, 3],
        "regressor__reg__alpha": [0.01, 0.1, 1, 10, 100],
    },
    cv=5,
    scoring="neg_root_mean_squared_error",
)
grid.fit(X_train, y_train)
print("Best hyperparameters:", grid.best_params_)
print("CV RMSE:", -grid.best_score_)

# Stage 10: one final look at the test set (predictions come back in original units)
y_pred = grid.predict(X_test)
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2 :", r2_score(y_test, y_pred))

# Stage 11: save the whole fitted pipeline, preprocessing included
joblib.dump(grid.best_estimator_, "regression_model.joblib")
loaded = joblib.load("regression_model.joblib")
new_predictions = loaded.predict(X_test.head())     # raw columns in, original units out
```

You do not need to memorize either sketch. Their point is the shape: **split first, fit every transformation on training data, tune with cross-validation, touch the test set once, and ship the whole pipeline.**

---

## 11. Common Mistakes to Avoid

* **Exploring or transforming before the split.** Looking at the test rows, or imputing, encoding, or scaling on the full dataset, lets the test set influence training.
* **Fitting on the test set.** Always `transform`, never `fit`, on test data.
* **Evaluating on the transformed scale** and comparing that number with models trained without a transformation.
* **Forgetting to save the transformation parameters** (such as $\lambda$), which makes the inverse step impossible.
* **Tuning on the test set.** Use cross-validation for every tuning decision, and look at the test set once.
* **Regularizing unscaled features.** The penalty then depends on the units you happened to choose.
* **Judging by $R^2$ alone.** Check the other metrics and the residual plots.
* **Skipping the baseline.** Without it, you cannot tell whether a complex model is actually better.
* **Deploying only the model.** Without its preprocessing, new data will not match what the model was trained on.
* **Treating deployment as the end.** Without monitoring, a model can quietly go wrong for months.

---

## 12. Where to Go Next

Work through the notebooks in order:

1. `Guide01`: transform the target and build a first end-to-end model.
2. `Guide02`: a realistic workflow with cleaning, EDA, assumption checks, and pipelines.
3. `Guide03`: encoding, train-test splitting, and scaling in depth.
4. `Guide04`: polynomial regression and grid search.
5. `Guide05`: cross-validation, regularization (Lasso and Ridge), and `GridSearchCV`.

Then continue with `01_Classification/`, where the same workflow applies with a categorical target.
