"""Sequence-free baseline models for Phase 7 comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class BaselineOutputs:
    linear_pred: np.ndarray
    tree_pred: np.ndarray
    mlp_pred: np.ndarray


def _feature_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = [
        c
        for c in df.columns
        if c.startswith("norm_") or c in {"candidate_limit", "current_limit", "candidate_delta", "candidate_delta_percent"}
    ]
    cat = [c for c in ["parameter", "direction", "tighten_or_loosen", "condition_id", "test_mode"] if c in df.columns]
    return num, cat


def _build_X(df: pd.DataFrame) -> pd.DataFrame:
    num, cat = _feature_cols(df)
    use = num + cat
    return df[use].copy()


def train_and_predict_baselines(
    *,
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    target_col: str = "target_score",
    random_state: int = 7,
) -> BaselineOutputs:
    num, cat = _feature_cols(train_df)
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ]
    )
    X_tr = _build_X(train_df)
    y_tr = train_df[target_col].to_numpy(dtype=float)
    X_pd = _build_X(pred_df)

    lin = Pipeline([("pre", pre), ("m", LinearRegression())])
    lin.fit(X_tr, y_tr)
    lin_pred = lin.predict(X_pd)

    tree = Pipeline([("pre", pre), ("m", HistGradientBoostingRegressor(random_state=random_state, max_depth=4, max_iter=120))])
    tree.fit(X_tr, y_tr)
    tree_pred = tree.predict(X_pd)

    mlp = Pipeline(
        [
            ("pre", pre),
            (
                "m",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    random_state=random_state,
                    max_iter=40,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )
    mlp.fit(X_tr, y_tr)
    mlp_pred = mlp.predict(X_pd)

    return BaselineOutputs(
        linear_pred=np.asarray(lin_pred, dtype=float),
        tree_pred=np.asarray(tree_pred, dtype=float),
        mlp_pred=np.asarray(mlp_pred, dtype=float),
    )
