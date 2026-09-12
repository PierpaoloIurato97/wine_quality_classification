import itertools
from typing import Literal

import numpy as np

import config
import utils
from model import WineQualityClassifier


def evaluate_accuracy(
    model: WineQualityClassifier, X: np.ndarray, y: np.ndarray
) -> float:
    """Calcola l'accuracy del modello."""
    pred = model.predict(X)
    return float((pred == y).sum() / len(y))


def grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> WineQualityClassifier:
    """Cerca la migliore combinazione di iperparametri sul validation set."""
    C_values = [1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    gamma_values: list[float | Literal["auto", "scale"]] = [
        "scale",
        0.001,
        0.003,
        0.005,
        0.008,
        0.01,
        0.015,
        0.02,
        0.03,
        0.05,
    ]

    best_accuracy = 0.0
    best_params: dict = {}
    best_model: WineQualityClassifier | None = None

    combinations = list(itertools.product(C_values, gamma_values))
    total = len(combinations)

    for i, (C, gamma) in enumerate(combinations, 1):
        try:
            model = WineQualityClassifier()
            model.fit(X_train, y_train, C=C, gamma=gamma, verbose=False)

            accuracy = evaluate_accuracy(model, X_val, y_val)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"C": C, "GAMMA": gamma}
                best_model = model
                print(
                    f"[{i}/{total}] Nuova migliore: "
                    f"accuracy={accuracy:.4f} | "
                    f"C={C}, gamma={gamma}"
                )
        except Exception:
            continue

    print(f"\nMigliori iperparametri: {best_params}")
    print(f"Migliore validation accuracy: {best_accuracy * 100:.2f}%")

    if best_model is None:
        raise RuntimeError("Nessuna combinazione di iperparametri ha funzionato.")

    return best_model


def train() -> None:
    # Carica dati di training
    df_train = utils.read_csv("winequality-red-train-standardized")
    X_train, y_train = utils.split_df(df_train)

    print(
        f"Vini buoni: {(y_train == 1).sum()} | " f"Vini cattivi: {(y_train == 0).sum()}"
    )

    # Carica dati di validazione per il grid search
    df_val = utils.read_csv("winequality-red-validation-standardized")
    X_val, y_val = utils.split_df(df_val)

    n_comb = 10 * 10
    print(f"Grid search su {n_comb} combinazioni di iperparametri...\n")
    model = grid_search(X_train, y_train, X_val, y_val)

    utils.save_model(model, "wine_quality_model")
    print("Modello salvato.")
