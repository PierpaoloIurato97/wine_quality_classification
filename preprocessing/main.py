import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared import DATA_DIR, ensure_dir_exists


def add_label_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'quality' not in df.columns:
        raise ValueError("Column 'quality' not found in raw data.")

    df['label'] = (df['quality'] > 5).astype(int)
    return df


def save_processed_data(df: pd.DataFrame):
    df.to_csv(os.path.join(DATA_DIR, 'winequality-red-processed.csv'),
              index=False, sep=';')


if __name__ == '__main__':
    ensure_dir_exists(DATA_DIR)

    df = pd.read_csv(os.path.join(DATA_DIR, 'winequality-red.csv'), sep=';')
    df = add_label_column(df)
    save_processed_data(df)
