import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

import config
import utils
from model import WineQualityClassifier


def _bayesian_optimize(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    """
    Runs Bayesian Optimisation via Optuna's TPE sampler to find the best
    (C, gamma) hyperparameters for an RBF-kernel SVM.

    Optuna's Tree-structured Parzen Estimator (TPE) builds a surrogate model
    over the objective function and uses it to decide which hyperparameters to
    try next, balancing exploration and exploitation automatically — the same
    Bayesian optimisation strategy as BayesSearchCV.

    A StratifiedKFold splitter is used inside the objective so that every fold
    preserves the class balance of the full training set.

    Returns (best_C, best_gamma, best_cv_accuracy).
    """

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float(
            "C", config.OPTUNA_C_BOUNDS[0], config.OPTUNA_C_BOUNDS[1], log=True
        )
        gamma = trial.suggest_float(
            "gamma",
            config.OPTUNA_GAMMA_BOUNDS[0],
            config.OPTUNA_GAMMA_BOUNDS[1],
            log=True,
        )
        svc = SVC(kernel="rbf", C=C, gamma=gamma)
        scores = cross_val_score(svc, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        return float(scores.mean())

    # Suppress Optuna's per-trial logging for a cleaner training output.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=config.OPTUNA_N_ITER)

    best_C = float(study.best_params["C"])
    best_gamma = float(study.best_params["gamma"])
    best_cv_acc = float(study.best_value)

    return best_C, best_gamma, best_cv_acc


def train() -> None:
    """
    Entry point for the training stage of the pipeline.

    Loads the standardized training set, finds the best hyperparameters
    (C, gamma) via Bayesian Optimisation with stratified K-Fold
    Cross-Validation, trains the final model on the full training set,
    and persists it to disk so that the evaluation step can load and
    benchmark it.
    """
    df_train = utils.read_csv("winequality-red-train-standardized")
    X_train, y_train = utils.split_df(df_train)

    best_C, best_gamma, best_cv_acc = _bayesian_optimize(X_train, y_train)

    print(f"\nOptimisation complete")
    print(f"Best hyperparameters: C={best_C:.4f}, gamma={best_gamma:.6f}")
    print(f"CV Accuracy: {best_cv_acc * 100:.2f}%")

    model = WineQualityClassifier()
    model.fit(X_train, y_train, C=best_C, gamma=best_gamma)

    pred = model.predict(X_train)
    train_accuracy = float((pred == y_train).sum() / len(y_train))
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")

    utils.save_model(model, "wine_quality_model")
    print("Model saved successfully.")
