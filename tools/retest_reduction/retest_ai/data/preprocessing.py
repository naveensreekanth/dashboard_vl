import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from ..config.settings import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ALL_MODEL_FEATURES,
    TARGET_COL,
    TARGET_MAPPING
)
from .validation import assert_no_leakage_in_feature_matrix, ValidationError


def cast_to_string_df(X):
    """Safely cast categorical inputs to string array to avoid dtype mismatches."""
    if isinstance(X, pd.DataFrame):
        return X.astype(str)
    return np.array(X, dtype=str)


def cast_to_float_df(X):
    """Cast numerical inputs to float. Does not invent replacement values."""
    if isinstance(X, pd.DataFrame):
        converted = X.apply(pd.to_numeric, errors="coerce")
        if converted.isnull().any().any():
            bad = converted.columns[converted.isnull().any()].tolist()
            raise ValidationError(f"Non-numeric or missing values in numerical features: {bad}")
        return converted.values
    arr = np.array(X, dtype=float)
    if np.isnan(arr).any():
        raise ValidationError("Non-numeric or missing values in numerical features.")
    return arr


def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for categorical and numerical features
    with automated type normalization.
    """
    cat_pipe = Pipeline([
        ("to_str", FunctionTransformer(cast_to_string_df)),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    num_pipe = Pipeline([
        ("to_float", FunctionTransformer(cast_to_float_df)),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
            ("num", num_pipe, NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_xy(
    df: pd.DataFrame,
    is_inference: bool = False
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Extracts the feature DataFrame X (using the strict feature whitelist) and target array y.
    Leakage and post-retest columns cannot enter X.
    """
    missing = [col for col in ALL_MODEL_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot prepare X: Missing features {missing}")

    assert_no_leakage_in_feature_matrix(ALL_MODEL_FEATURES)

    X = df[ALL_MODEL_FEATURES].copy()
    assert_no_leakage_in_feature_matrix(X.columns)

    for col in CATEGORICAL_FEATURES:
        if X[col].isnull().any():
            raise ValidationError(f"Missing values in categorical feature {col}. Data was not altered.")
        X[col] = X[col].astype(str)

    for col in NUMERICAL_FEATURES:
        numeric = pd.to_numeric(X[col], errors="coerce")
        if numeric.isnull().any():
            raise ValidationError(
                f"Missing or non-numeric values in {col}. Data was not silently filled."
            )
        X[col] = numeric.astype(float)

    y = None
    if not is_inference and TARGET_COL in df.columns:
        y_raw = df[TARGET_COL].astype(str).str.strip()
        y = y_raw.map(TARGET_MAPPING).values
        if np.isnan(y).any():
            raise ValueError(f"Unmapped target values found in {TARGET_COL}")
        y = y.astype(int)

    return X, y


def get_feature_names_after_preprocessing(preprocessor: ColumnTransformer) -> List[str]:
    """
    Retrieves the transformed feature names from the ColumnTransformer.
    """
    try:
        if hasattr(preprocessor, "get_feature_names_out"):
            return list(preprocessor.get_feature_names_out())
    except Exception:
        pass

    feature_names = []
    transformers = getattr(preprocessor, "transformers_", getattr(preprocessor, "transformers", []))
    for item in transformers:
        if len(item) >= 3:
            name, transformer, cols = item[0], item[1], item[2]
            if name == "cat":
                ohe = transformer.named_steps["ohe"] if hasattr(transformer, "named_steps") else transformer
                if hasattr(ohe, "get_feature_names_out"):
                    cat_names = ohe.get_feature_names_out(cols)
                    feature_names.extend(cat_names.tolist())
                else:
                    feature_names.extend(cols)
            elif name == "num":
                feature_names.extend(cols)
    return feature_names if feature_names else ALL_MODEL_FEATURES
