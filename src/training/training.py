import itertools

import numpy as np

import utils
from model import WineQualityClassifier


def evaluate_accuracy(
    model: WineQualityClassifier, X: np.ndarray, y: np.ndarray
) -> float:
    pred = model.predict(X)
    return float((pred == y).sum() / len(y))


def grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> WineQualityClassifier:
    C_values = [1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    gamma_values = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07]

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
    df_train = utils.read_csv("winequality-red-train-standardized")
    X_train, y_train = utils.split_df(df_train)

    df_val = utils.read_csv("winequality-red-validation-standardized")
    X_val, y_val = utils.split_df(df_val)

    model = grid_search(X_train, y_train, X_val, y_val)

    utils.save_model(model, "wine_quality_model")
    print("Model saved successfully.")
