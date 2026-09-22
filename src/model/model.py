import numpy as np
from sklearn.svm import SVC


class WineQualityClassifier:
    """
    Thin wrapper around scikit-learn's SVC that encapsulates the classifier
    used throughout the training and evaluation pipeline.

    Keeping the model behind this interface decouples the rest of the codebase
    from the specific sklearn API, making it straightforward to swap the
    underlying algorithm in the future.
    """

    def __init__(self):
        """Initialises the classifier with no trained model yet."""
        self.model: SVC | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, C: float, gamma: float) -> None:
        """
        Trains an RBF-kernel SVM on the provided data.

        Called by grid_search in the training module once per hyperparameter
        combination. C controls the regularisation strength and gamma the
        reach of each training example's influence.
        """
        self.model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
        )
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns class predictions for the given samples.

        Used both during grid search (to score each hyperparameter combination
        on the validation set) and during final evaluation on train/val/test.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Returns decision function scores (distance to the hyperplane) for the samples.
        Useful for calculating ranking metrics like ROC AUC.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.decision_function(X)
