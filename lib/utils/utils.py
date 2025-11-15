import os

import pandas as pd

DATA_DIR = 'data'
PLOTS_DIR = 'plots'


def ensure_dir_exists(dir_path: str):
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Data directory '{dir_path}' does not exist.")


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep=';')


def print_descriptive_statistics(df: pd.DataFrame):
    print(df.describe(), '\n')
