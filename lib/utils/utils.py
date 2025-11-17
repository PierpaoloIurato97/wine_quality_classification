import os
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import torch

DATA_DIR = 'data'
PLOTS_DIR = 'plots'
MODELS_DIR = 'models'


def ensure_dir_exists(dir_path: str):
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Data directory '{dir_path}' does not exist.")


def read_csv(file_name: str) -> pd.DataFrame:
    ensure_dir_exists(DATA_DIR)
    return pd.read_csv(os.path.join(DATA_DIR, f'{file_name}.csv'), sep=';')


def save_csv(df: pd.DataFrame, file_name: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, f'{file_name}.csv'), index=False, sep=';')


def print_descriptive_statistics(df: pd.DataFrame):
    print(df.describe(), '\n')


def make_plot(title: str, xlabel: str, ylabel: str, file_name: str, plot: Callable):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(PLOTS_DIR, f'{file_name}.png'))
    plt.close()


def save_model(model: torch.nn.Module, model_name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(MODELS_DIR, f'{model_name}.pth')
    )


def load_model(model: torch.nn.Module, model_name: str):
    ensure_dir_exists(MODELS_DIR)
    model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, f'{model_name}.pth'))
    )
