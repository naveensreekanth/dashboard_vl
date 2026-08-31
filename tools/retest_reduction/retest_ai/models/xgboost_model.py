from sklearn.pipeline import Pipeline
import xgboost as xgb
from ..data.preprocessing import create_preprocessing_pipeline
from ..config.settings import RANDOM_SEED

def build_xgboost_model(
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = RANDOM_SEED
) -> Pipeline:
    """
    Builds the XGBoost primary model pipeline.
    """
    preprocessor = create_preprocessing_pipeline()
    classifier = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="logloss",
        random_state=random_state
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
