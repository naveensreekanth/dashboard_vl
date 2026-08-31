from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from ..data.preprocessing import create_preprocessing_pipeline
from ..config.settings import RANDOM_SEED

def build_logistic_model(
    C: float = 1.0,
    random_state: int = RANDOM_SEED
) -> Pipeline:
    """
    Builds the Logistic Regression baseline model pipeline.
    """
    preprocessor = create_preprocessing_pipeline()
    classifier = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
