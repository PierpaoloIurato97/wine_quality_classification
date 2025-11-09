import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared import (DATA_DIR, PLOTS_DIR, ensure_dir_exists, load_data,
                    print_descriptive_statistics)


def describe_and_plot_processed_data(df: pd.DataFrame):
    if 'label' not in df.columns:
        raise ValueError("Column 'label' not found in processed data.")

    print("Descriptive statistics for processed data:", '\n')
    print_descriptive_statistics(df)

    print("Class distribution in processed data:")
    print(df['label'].value_counts())

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.hist(df['label'], bins=2, edgecolor='black')
    plt.title('Processed Wine Quality Distribution')
    plt.xlabel('Quality')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(
        PLOTS_DIR, 'processed_wine_quality_distribution.png'))
    plt.close()


ensure_dir_exists(DATA_DIR)

df = load_data(os.path.join(DATA_DIR, 'winequality-red-processed.csv'))
describe_and_plot_processed_data(df)
