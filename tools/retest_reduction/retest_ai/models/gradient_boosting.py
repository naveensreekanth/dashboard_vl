from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from ..data.preprocessing import create_preprocessing_pipeline
from ..config.settings import RANDOM_SEED

def build_gradient_boosting_model(
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    random_state: int = RANDOM_SEED
) -> Pipeline:
    """
    Builds the Scikit-Learn GradientBoostingClassifier challenger pipeline.
    """
    preprocessor = create_preprocessing_pipeline()
    classifier = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=random_state
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
