import os
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import torch

DATA_DIR = 'data'
PLOTS_DIR = 'plots'
MODELS_DIR = 'models'
MODEL_VERSION = '0.1.0'


def ensure_dir_exists(dir_path: str):
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Data directory '{dir_path}' does not exist.")


def read_csv(file_name: str) -> pd.DataFrame:
    data_dir_path = os.path.join('..', DATA_DIR)

    ensure_dir_exists(data_dir_path)
    return pd.read_csv(
        os.path.join(data_dir_path, f'{file_name}.csv'), sep=';'
    )


def save_csv(df: pd.DataFrame, file_name: str):
    data_dir_path = os.path.join('..', DATA_DIR)

    ensure_dir_exists(data_dir_path)
    df.to_csv(
        os.path.join(data_dir_path, f'{file_name}.csv'), index=False, sep=';'
    )


def print_descriptive_statistics(df: pd.DataFrame):
    print(df.describe(), '\n')


def make_plot(title: str, file_name: str, plot: Callable, xlabel: str = '', ylabel: str = ''):
    plot()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    save_plot(file_name)


def save_plot(file_name: str):
    plot_dir_path = os.path.join('..', PLOTS_DIR)

    os.makedirs(plot_dir_path, exist_ok=True)

    plt.savefig(os.path.join(plot_dir_path, f'{file_name}.png'))
    plt.close()


def save_model(model: torch.nn.Module, model_name: str):
    model_dir_path = os.path.join('..', MODELS_DIR)

    os.makedirs(model_dir_path, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(model_dir_path, f'{model_name}_{MODEL_VERSION}.pth')
    )


def load_model(model: torch.nn.Module, model_name: str):
    model_dir_path = os.path.join('..', MODELS_DIR)

    ensure_dir_exists(model_dir_path)
    model.load_state_dict(torch.load(
        os.path.join(model_dir_path, f'{model_name}_{MODEL_VERSION}.pth'))
    )


def train_mode(model: torch.nn.Module) -> None:
    torch.set_grad_enabled(True)
    model.train()


def eval_mode(model: torch.nn.Module) -> None:
    torch.set_grad_enabled(False)
    model.eval()


def split_df_for_inference(df: pd.DataFrame, label_col_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    input = torch.from_numpy(
        df.drop(columns=[label_col_name]).to_numpy()
    ).float()
    labels = torch.from_numpy(
        df[label_col_name].to_numpy()
    ).float()

    return input, labels
