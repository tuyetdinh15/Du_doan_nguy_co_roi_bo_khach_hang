
from sklearn.linear_model import LogisticRegression


def create_model(C=1.0, class_weight=None, max_iter=2000):
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        solver="liblinear",
        random_state=42
    )
