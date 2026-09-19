import itertools

import numpy as np

import utils
from model import WineQualityClassifier


def evaluate_accuracy(
    model: WineQualityClassifier, X: np.ndarray, y: np.ndarray
) -> float:
    """
    Computes the fraction of correctly classified samples for a given split.

    Serves as the scoring metric inside grid_search so that each hyperparameter
    combination can be ranked and the best one selected.
    """
    pred = model.predict(X)
    return float((pred == y).sum() / len(y))


def grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> WineQualityClassifier:
    """
    Selects the best SVM hyperparameters by exhaustive search over a fixed
    grid of C and gamma values.

    For each combination the model is trained on the training set and scored
    on the validation set, keeping the combination that yields the highest
    accuracy. The validation set is used here — not the test set — to avoid
    biasing the final performance estimate.
    """
    C_values = [
        1.0,
        5.0,
        10.0,
        50.0,
        100.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        700.0,
        800.0,
        1000.0,
        1500.0,
        2000.0,
        3000.0,
        5000.0,
    ]
    gamma_values = [
        0.001,
        0.003,
        0.005,
        0.008,
        0.01,
        0.012,
        0.014,
        0.015,
        0.016,
        0.018,
        0.02,
        0.025,
        0.03,
        0.05,
        0.07,
    ]

    best_accuracy = 0.0
    best_params: dict = {}
    best_model: WineQualityClassifier | None = None

    combinations = list(itertools.product(C_values, gamma_values))

    for C, gamma in combinations:
        model = WineQualityClassifier()
        model.fit(X_train, y_train, C=C, gamma=gamma)

        accuracy = evaluate_accuracy(model, X_val, y_val)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {"C": C, "gamma": gamma}
            best_model = model

    print(f"Best hyperparameters: {best_params}")
    print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")

    if best_model is None:
        raise RuntimeError("No hyperparameter combination produced a valid model.")

    return best_model


def train() -> None:
    """
    Entry point for the training stage of the pipeline.

    Loads the standardized train and validation sets, runs grid search to find
    the optimal hyperparameters, and persists the resulting model to disk so
    that the evaluation step can load and benchmark it.
    """
    df_train = utils.read_csv("winequality-red-train-standardized")
    X_train, y_train = utils.split_df(df_train)

    df_val = utils.read_csv("winequality-red-validation-standardized")
    X_val, y_val = utils.split_df(df_val)

    model = grid_search(X_train, y_train, X_val, y_val)

    utils.save_model(model, "wine_quality_model")
    print("Model saved successfully.")
