import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared import (DATA_DIR, PLOTS_DIR, ensure_dir_exists, load_data,
                    print_descriptive_statistics)


def describe_and_plot_raw_data(df: pd.DataFrame):
    if 'quality' not in df.columns:
        raise ValueError("Column 'label' not found in raw data.")

    print("Descriptive statistics for raw data:", '\n')
    print_descriptive_statistics(df)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.hist(df['quality'], bins=10, edgecolor='black')
    plt.title('Wine Quality Distribution')
    plt.xlabel('Quality')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(PLOTS_DIR, 'wine_quality_distribution.png'))
    plt.close()


ensure_dir_exists(DATA_DIR)

df = load_data(os.path.join(DATA_DIR, 'winequality-red.csv'))
describe_and_plot_raw_data(df)
