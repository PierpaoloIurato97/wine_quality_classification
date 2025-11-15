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

    utils.make_plot(
        title='Processed Wine Quality Distribution',
        xlabel='Quality',
        ylabel='Frequency',
        file_name='processed_wine_quality_distribution',
        plot=lambda: plt.hist(df['label'], bins=2, edgecolor='black')
    )


df = utils.read_csv('winequality-red-processed')
describe_and_plot_processed_data(df)
