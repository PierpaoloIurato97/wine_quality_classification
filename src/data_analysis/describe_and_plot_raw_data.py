import matplotlib.pyplot as plt

import utils


def describe_and_plot_raw_data():
    df = utils.read_csv('winequality-red')

    if 'quality' not in df.columns:
        raise ValueError("Column 'label' not found in raw data.")

    print("Descriptive statistics for raw data:", '\n')
    utils.print_descriptive_statistics(df)

    utils.make_plot(
        title='Wine Quality Distribution',
        xlabel='Quality',
        ylabel='Frequency',
        file_name='wine_quality_distribution',
        plot=lambda: plt.hist(df['quality'], bins=5, edgecolor='black')
    )
