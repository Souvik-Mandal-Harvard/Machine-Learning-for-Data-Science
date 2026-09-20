# Guide00: Linear Regression Basic Workflow (Level 01)

This guide turns the ideas from the [Introduction to Supervised Machine Learning](../Guide00_Introduction-to-Supervised-Machine-Learning.md) into a concrete, repeatable recipe for building a linear regression model. Read it before opening the notebooks: each notebook (`Guide01`–`Guide05`) practices one or more steps of this workflow in depth.

By the end of this guide, you should be able to:
* list the steps of a basic linear regression workflow, in the right order,
* explain why the train-test split comes before any learned transformation,
* explain why predictions must be converted back to original units before evaluation,
* find which notebook practices each step.

---

## 1. The Workflow at a Glance

| # | Step | Practiced in |
| :-- | :--- | :--- |
| 0 | Clean and explore the data | `Guide02` |
| 1 | Define target ($y$) and features ($X$) | `Guide01`, `Guide02` |
| 2 | Train-test split | `Guide01`, `Guide02`, `Guide03` |
| 3 | Encode categorical features | `Guide02`, `Guide03` |
| 4 | Feature engineering (polynomials, interactions) | `Guide01`, `Guide04` |
| 5 | Target transformation | `Guide01` |
| 6 | Feature scaling | `Guide01`, `Guide03` |
| 7 | Fit the model | all notebooks |
| 8 | Predict on the test set | all notebooks |
| 9 | Inverse-transform the predictions | `Guide01` |
| 10 | Evaluate with metrics | `Guide01`, `Guide02` |
| 11 | Residual diagnostics | `Guide02` |

Not every project needs every step. Steps 3–6 apply only when your data calls for them (categorical columns, curved relationships, a skewed target, features on very different scales).

---

## 2. The One Rule: Learn From Training Data Only

The test set stands in for future, unseen data. If any information from it leaks into training, your test score is no longer an honest estimate of real-world performance. This is called **data leakage**.

Anything that *learns something from data* must be fit on the training set only, then applied to the test set:

* the categories an encoder discovers,
* the mean and standard deviation a scaler computes,
* the $\lambda$ found by a Box-Cox transformation.

In scikit-learn terms: call `fit` / `fit_transform` on the training data, and only `transform` on the test data. This is why the split comes first.

---

## 3. The Steps in Detail

### Step 0. Clean and Explore the Data
Handle missing values, duplicates, and inconsistent labels, then explore the data with summary statistics and plots. Look at the distribution of the target, the relationship between each numeric feature and the target (is it roughly linear?), and correlations *among* features (severe multicollinearity makes coefficients unstable).

### Step 1. Define Variables
Choose the target variable ($y$, a continuous quantity) and the feature variables ($X$). Decide on the success metric now, before you see any results.

### Step 2. Train-Test Split
Split into `X_train`, `X_test`, `y_train`, `y_test` (for example, 80/20) **before** applying any transformation that learns from data. Fix `random_state` so your results are reproducible.

### Step 3. Encode Categorical Features
Linear regression needs numbers. Convert nominal categories with one-hot encoding, dropping one level per category so the coefficients stay uniquely interpretable. Fit the encoder on `X_train`, then apply it to `X_test`.

### Step 4. Feature Engineering (Polynomials and Interactions)
To capture non-linear relationships while keeping a linear model, create polynomial or interaction terms.
* **Rule:** define these on `X_train`, then apply the same transformation to `X_test`.
* `PolynomialFeatures` learns nothing from the data, so it cannot leak by itself, but following the same fit-on-train habit as every other step is the safer default.
* Watch the column count: degree-2 terms grow quickly as you add features, which raises the risk of overfitting.

### Step 5. Target Variable Transformation
Linear regression does not require the *target* to be normal. Normally distributed *residuals* matter mainly for inference (confidence intervals and p-values). Still, transforming a strongly skewed target often stabilizes the error variance and improves the fit.
* Check the distribution of `y_train` (histogram, `stats.normaltest`).
* If it is skewed, apply a transformation such as log, square root, or Box-Cox to `y_train` only.
* **Crucial:** save the transformation parameters (like the Box-Cox $\lambda$) so you can invert it in Step 9.

### Step 6. Feature Scaling
Standardize or normalize the features.
* Plain least-squares predictions do not change with feature scale, but scaling matters for **regularization** (Ridge/Lasso penalize all coefficients equally, so features must be comparable), for gradient-based solvers, and for comparing coefficient sizes.
* Fit the scaler (e.g., `StandardScaler`) on `X_train`; use it to transform both `X_train` and `X_test`.

### Step 7. Fit the Model
Fit the linear regression on the fully processed training data: the encoded, engineered, scaled `X_train` and the transformed `y_train`. This is where the algorithm minimizes the loss and estimates the parameters ($\beta_0, \beta_1, \dots, \beta_p$).

### Step 8. Predict
Pass the processed `X_test` (transformed with the *training-fitted* tools from Steps 3–6) to the model to get predictions $\hat{y}$.

### Step 9. Inverse-Transform the Predictions
If you transformed the target in Step 5, these predictions are still on the transformed scale (for example, the Box-Cox scale). Use the saved parameters to convert them back to the original units of $y$ (dollars, not log-dollars).

### Step 10. Evaluate
Compare the **inverse-transformed** predictions against the **original, untouched** `y_test`, using several metrics: MAE, RMSE, and $R^2$. Metrics computed on the transformed scale are not comparable to metrics from a model without a transformation.

### Step 11. Residual Diagnostics
Validate the model's assumptions by analyzing residuals ($y - \hat{y}$):
* **Linearity and constant variance (homoscedasticity):** residuals vs. fitted values should show no curve and no funnel.
* **Normality of residuals:** check with a Q-Q plot (matters mainly for inference).
* Compare training and test errors: a much lower training error signals overfitting.

---

## 4. The Workflow as Code (Sketch)

A minimal illustration of the fit-on-train discipline. It assumes `X` is already numeric (Step 3 done) and `y > 0` (Box-Cox requirement):

```python
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Step 2: split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: feature engineering (fit on train, apply to both)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_p = poly.fit_transform(X_train)
X_test_p = poly.transform(X_test)

# Step 6: scaling (fit on train only)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_p)
X_test_s = scaler.transform(X_test_p)

# Step 5: target transformation (lambda estimated from train only)
y_train_bc, lam = stats.boxcox(y_train)

# Steps 7-8: fit and predict
model = LinearRegression().fit(X_train_s, y_train_bc)
y_pred_bc = model.predict(X_test_s)

# Step 9: back to original units
y_pred = inv_boxcox(y_pred_bc, lam)

# Step 10: evaluate against the untouched y_test
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2 :", r2_score(y_test, y_pred))
```

Writing every step by hand is the best way to learn what each one does, but it is error-prone. In practice, you wrap the preprocessing and the model into a scikit-learn **`Pipeline`** (with a `ColumnTransformer` for mixed column types), which applies the steps in the right order and fits them on training data automatically. `Guide02` introduces this, and it becomes essential for cross-validation in `Guide05`.

---

## 5. Common Mistakes to Avoid

* **Scaling or encoding before the split.** The scaler or encoder then "sees" the test data.
* **Fitting on the test set.** Always `transform`, never `fit`, on test data.
* **Evaluating on the transformed scale** and comparing that number with models trained without a transformation.
* **Forgetting to save the transformation parameters** (such as $\lambda$), which makes the inverse step impossible.
* **Judging by $R^2$ alone.** Check the other metrics and the residual plots.
* **Reusing the test set for repeated tuning.** Use cross-validation for that (`Guide05`).

---

## 6. Where to Go Next

Work through the notebooks in order:

1. `Guide01`: transform the target and build a first end-to-end model.
2. `Guide02`: a realistic workflow with cleaning, EDA, assumption checks, and pipelines.
3. `Guide03`: encoding, train-test splitting, and scaling in depth.
4. `Guide04`: polynomial regression and grid search.
5. `Guide05`: cross-validation, regularization (Lasso and Ridge), and `GridSearchCV`.

This Level 01 workflow evaluates a model on a single train-test split. Cross-validation and hyperparameter tuning, which make that evaluation more reliable, come at the end of the series.
