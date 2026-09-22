"""Shared, reusable code for the Ames Housing regression guides (Guide01-Guide06).

This module grows across the guide series: each function's docstring names the guide that introduced it. Once a guide's recap cell depends on a function's signature, later guides may only ADD new functions, or new optional keyword arguments with defaults -- never change what an earlier guide already relies on. Git history is this file's version record; see the "Stage 6 to-do list" and "What we hand to the next guide" cells in each notebook for what changed and why.

Import it from a Guide0N notebook running in 00_Regression/ (the notebooks'
working directory) with:

    from pipeline.ames_workflow import load_ames, clean_ames, split_ames
    from pipeline.ames_workflow import add_engineered_features, build_preprocessor

Requires scikit-learn 1.1 or newer (``OneHotEncoder``'s ``min_frequency``).
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, PolynomialFeatures, StandardScaler,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Ames_Housing_Sales.csv"

# The ten columns below all grade a feature on the same scale, worst to best:
# no such feature < poor < fair < typical/average < good < excellent. Six of
# them (Exter*, HeatingQC, KitchenQual, Garage*) never actually take the
# "None" value in this file -- every house here has walls, heat, a kitchen,
# and a garage -- but sharing one definition keeps the code simple and stays
# correct if a future, more complete Ames extract includes such rows too.
# A few other columns (LotShape, Functional, GarageFinish, ...) are also
# mildly ordinal, on their own, different scales; Guide02 treats those as
# nominal to keep the notebook focused, and flags it as a simplification.
QUALITY_SCALE = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]


def load_ames(path=DATA_PATH):
    """Guide01, Stage 2 (Understand the data). Read the raw Ames Housing CSV.

    ``keep_default_na=False`` is deliberate, not an oversight: every column
    pandas would otherwise report as having "missing" values here is a
    genuine category whose label happens to be the literal text "None" (no
    pool, no alley, no basement, ...) -- one of pandas' default NA sentinel
    strings. Reading with pandas' default NA parsing silently turns 8,657
    valid category labels into NaN across 11 columns, none of which are
    actually missing. See Guide01's Stage 2/3 sections for the demonstration.
    """
    return pd.read_csv(path, keep_default_na=False)


def clean_ames(df):
    """Guide01, Stage 3 (Clean the data). Fixes that do not learn from the
    data, so they are safe to apply before the train-test split (Stage 4).

    Two fixes:

    1. ``MSSubClass`` is stored as a number (20, 60, 120, ...) but it names a
       building-type category, not a quantity -- a wrong data type, in
       Guide00's Stage 3 sense. Cast it to text so it is encoded, not
       averaged, from Guide02 onward.
    2. Two houses with GrLivArea > 4,000 sq ft were sold as "Partial": new
       construction sold before it was finished, so their price does not
       reflect a typical, finished, arms-length sale the way the rest of the
       data does. They are removed, with the reason on record. Two other
       houses above 4,000 sq ft are ordinary "Normal"/"Abnorml" sales and are
       kept -- being large is not itself a reason to drop a row.
    """
    df = df.copy()

    df["MSSubClass"] = df["MSSubClass"].astype(str)

    is_partial_giant = (df["GrLivArea"] > 4000) & (df["SaleCondition"] == "Partial")
    df = df.loc[~is_partial_giant].reset_index(drop=True)

    return df


def split_ames(df, test_years=(2010,)):
    """Guide01, Stage 4 (Split). A time-based split, not a random one: train
    on sales through 2009, test on the most recent year alone.

    A random split would let training rows and test rows sit side by side in
    the same year, which is not how the model will actually be used: it will
    always be asked about sales that have not happened yet. Splitting by year
    matches that, and means the single Stage 10 look at the test set (Guide06)
    doubles as a first check for drift (Stage 12) against a genuinely future
    year, instead of needing a second, separate demonstration.

    The test set returned here must not be explored, transformed-and-fit-on,
    or evaluated against until Guide06.
    """
    test_mask = df["YrSold"].isin(test_years)
    train_df, test_df = df.loc[~test_mask], df.loc[test_mask]

    X_train = train_df.drop(columns="SalePrice")
    y_train = train_df["SalePrice"]
    X_test = test_df.drop(columns="SalePrice")
    y_test = test_df["SalePrice"]
    return X_train, X_test, y_train, y_test


def add_engineered_features(df):
    """Guide02, Stage 6.3 (Engineer features). Two domain-driven columns
    built from existing ones. Both are plain arithmetic -- they do not learn
    any statistic from the data -- so, unlike the steps in
    ``build_preprocessor``, they are safe to apply to training and test data
    the same way, independently, with no risk of leakage.

    * ``HouseAge``: how old the house was in the year it sold.
    * ``TotalSF``: total finished-and-basement square footage, combining
      three separately-measured area columns into one.
    """
    df = df.copy()
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    return df


def build_preprocessor(
    reference_df, rare_category_min_count=15, poly_cols=None, degree=2,
):
    """Guide02, Stage 6 (Transform the data). Returns an unfitted
    ``ColumnTransformer``: impute, then encode or scale, for every column of
    ``reference_df`` (which should already have ``clean_ames`` and
    ``add_engineered_features`` applied).

    ``reference_df`` is used only to read off column names and dtypes --
    metadata that does not depend on which rows are in it -- never to fit
    anything. Fit the returned object with ``.fit()`` / ``.fit_transform()``
    on training data only, then ``.transform()`` validation or test data with
    that same fitted object (Guide00 Rule 1).

    Three branches, one per column role:

    * **Numeric** columns: median imputation, then standardize.
    * **Ordinal** columns (``QUALITY_COLS``): most-frequent imputation, then
      map to integers in ``QUALITY_SCALE`` order.
    * **Nominal** columns (every other text column): most-frequent
      imputation, then one-hot encode. Categories seen fewer than
      ``rare_category_min_count`` times in training data are pooled into a
      single "infrequent" column instead of getting one of their own --
      learned from training data only, exactly like every other step here.

    Guide04, Stage 9.1 adds ``poly_cols`` and ``degree`` (both optional, and
    both ``None``/``2`` by default): when ``poly_cols`` is given, those
    numeric columns run through ``PolynomialFeatures(degree)`` -- which adds
    every column's power up to ``degree`` *and* every pairwise product
    between them, i.e. interaction terms, not just curvature -- before
    scaling; every other numeric column is imputed and scaled exactly as
    before. Leaving ``poly_cols`` at its default reproduces Guide02 and
    Guide03's preprocessor exactly, unchanged.
    """
    numeric_cols = list(reference_df.select_dtypes("number").columns)
    ordinal_cols = [c for c in QUALITY_COLS if c in reference_df.columns]
    nominal_cols = [
        c for c in reference_df.select_dtypes(exclude="number").columns
        if c not in ordinal_cols
    ]
    poly_cols = [c for c in (poly_cols or []) if c in numeric_cols]
    plain_numeric_cols = [c for c in numeric_cols if c not in poly_cols]

    plain_numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    poly_numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
    ])
    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[QUALITY_SCALE] * len(ordinal_cols))),
    ])
    nominal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=rare_category_min_count,
        )),
    ])

    branches = []
    if plain_numeric_cols:
        branches.append(("num", plain_numeric_pipe, plain_numeric_cols))
    if poly_cols:
        branches.append(("num_poly", poly_numeric_pipe, poly_cols))
    branches.append(("ord", ordinal_pipe, ordinal_cols))
    branches.append(("nom", nominal_pipe, nominal_cols))

    return ColumnTransformer(branches)
