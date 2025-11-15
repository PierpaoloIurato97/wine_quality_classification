import os

import matplotlib.pyplot as plt
import pandas as pd
import utils


def describe_and_plot_raw_data(df: pd.DataFrame):
    if 'quality' not in df.columns:
        raise ValueError("Column 'label' not found in raw data.")

    print("Descriptive statistics for raw data:", '\n')
    utils.print_descriptive_statistics(df)

    os.makedirs(utils.PLOTS_DIR, exist_ok=True)
    plt.hist(df['quality'], bins=10, edgecolor='black')
    plt.title('Wine Quality Distribution')
    plt.xlabel('Quality')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(utils.PLOTS_DIR, 'wine_quality_distribution.png'))
    plt.close()


utils.ensure_dir_exists(utils.DATA_DIR)

df = utils.load_data(os.path.join(utils.DATA_DIR, 'winequality-red.csv'))
describe_and_plot_raw_data(df)
