import os

import matplotlib.pyplot as plt
import pandas as pd
import utils


def describe_and_plot_processed_data(df: pd.DataFrame):
    if 'label' not in df.columns:
        raise ValueError("Column 'label' not found in processed data.")

    print("Descriptive statistics for processed data:", '\n')
    utils.print_descriptive_statistics(df)

    print("Class distribution in processed data:")
    print(df['label'].value_counts())

    os.makedirs(utils.PLOTS_DIR, exist_ok=True)
    plt.hist(df['label'], bins=2, edgecolor='black')
    plt.title('Processed Wine Quality Distribution')
    plt.xlabel('Quality')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(
        utils.PLOTS_DIR, 'processed_wine_quality_distribution.png'))
    plt.close()


utils.ensure_dir_exists(utils.DATA_DIR)

df = utils.load_data(os.path.join(
    utils.DATA_DIR, 'winequality-red-processed.csv'))
describe_and_plot_processed_data(df)
