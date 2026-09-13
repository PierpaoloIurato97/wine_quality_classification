import numpy as np
from sklearn.svm import SVC


class WineQualityClassifier:
    def __init__(self):
        self.model: SVC | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, C: float, gamma: float) -> None:
        self.model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
        )
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict(X)
