from typing import Literal

import numpy as np
from sklearn.svm import SVC

import config


class WineQualityClassifier:
    def __init__(self):
        self.model: SVC | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C: float = config.SVM_C,
        gamma: float | Literal["auto", "scale"] = config.SVM_GAMMA,
        verbose: bool = True,
    ) -> None:
        """Addestra il modello SVC con kernel RBF."""
        self.model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
        )
        self.model.fit(X, y)
        if verbose:
            print(f"Support vectors: {self.model.n_support_} (per classe)")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice la qualità."""
        if self.model is None:
            raise RuntimeError("Il modello non è stato addestrato.")
        return self.model.predict(X)
