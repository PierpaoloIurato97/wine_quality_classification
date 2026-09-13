import os
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def ensure_dir_exists(dir_path: str):
    """
    Guards every file I/O operation by asserting that the target directory
    exists before attempting to read or write.

    Raises an early, descriptive error instead of letting Python produce a
    cryptic OS-level exception mid-pipeline.
    """
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Data directory '{dir_path}' does not exist.")


def read_csv(file_name: str) -> pd.DataFrame:
    """
    Loads a semicolon-delimited CSV from the project data directory.

    All pipeline steps use this function to read their input, keeping the
    file-naming convention and separator in a single place.
    """
    ensure_dir_exists(config.DATA_DIR)
    return pd.read_csv(os.path.join(config.DATA_DIR, f"{file_name}.csv"), sep=";")


def save_csv(df: pd.DataFrame, file_name: str):
    """
    Persists a DataFrame as a semicolon-delimited CSV in the project data
    directory.

    Each preprocessing step writes its output here so that the next step
    can pick it up independently, allowing stages to be rerun in isolation.
    """
    ensure_dir_exists(config.DATA_DIR)
    df.to_csv(os.path.join(config.DATA_DIR, f"{file_name}.csv"), index=False, sep=";")


def ensure_col_exists(df: pd.DataFrame, col_name: str):
    """
    Validates that an expected column is present in a DataFrame before
    processing begins.

    Catches schema mismatches (e.g. reading the wrong file) early and raises
    a descriptive error rather than a confusing KeyError later.
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in df.")


def print_descriptive_statistics(df: pd.DataFrame):
    """
    Prints a summary statistics table (count, mean, std, quartiles) for all
    numeric columns in the DataFrame.

    Used by the data-description steps to give a quick statistical overview
    of both the raw and the processed datasets.
    """
    print(df.describe(), "\n")


def make_plot(
    title: str, file_name: str, plot: Callable, xlabel: str = "", ylabel: str = ""
):
    """
    Applies common formatting (title, axis labels) and delegates the actual
    drawing to a caller-supplied function, then saves the result to disk.

    This pattern avoids duplicating plt setup and teardown code across every
    individual plotting function in the data-analysis modules.
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plot()

    save_plot(file_name)


def save_plot(file_name: str):
    """
    Saves the current matplotlib figure to the project plots directory and
    closes it to free memory.

    Called exclusively by make_plot so all plots are written to the same
    output folder without each caller managing the path.
    """
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.PLOTS_DIR, f"{file_name}.png"))
    plt.close()


def save_model(model, model_name: str):
    """
    Serialises the trained classifier to a versioned joblib file in the
    models directory.

    Versioning the filename allows multiple model versions to coexist on disk,
    making it possible to roll back or compare across runs.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(
        model,
        os.path.join(config.MODELS_DIR, f"{model_name}_{config.MODEL_VERSION}.joblib"),
    )


def load_model(model_name: str):
    """
    Deserialises and returns the versioned model from disk.

    Used by the evaluation step to load the model that was produced by the
    training step without coupling the two stages directly.
    """
    ensure_dir_exists(config.MODELS_DIR)
    return joblib.load(
        os.path.join(config.MODELS_DIR, f"{model_name}_{config.MODEL_VERSION}.joblib"),
    )


def split_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Separates a DataFrame into a feature matrix X and a label vector y.

    Reused by every stage that needs to feed data into the model (training,
    grid search, evaluation), ensuring a consistent feature ordering.
    """
    X = df[config.FEATURES].to_numpy().astype(np.float64)
    y = df[config.LABEL].to_numpy().astype(int)
    return X, y


def calculate_accuracy(num_correct: int, dataset_len: int) -> float:
    """
    Returns the accuracy as a fraction rounded to four decimal places.

    Centralises the accuracy formula so all evaluation code produces
    identically formatted values.
    """
    return round(num_correct / dataset_len, 4)
