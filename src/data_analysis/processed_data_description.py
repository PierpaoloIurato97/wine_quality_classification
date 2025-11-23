import matplotlib.pyplot as plt

import utils


def describe_processed_data():
    df = utils.read_csv('winequality-red-with-label-standardized')

    utils.ensure_col_exists(df, 'label')

    print("Descriptive statistics for processed data:", '\n')
    utils.print_descriptive_statistics(df)

    utils.make_plot(
        title='Processed Wine Label Distribution',
        xlabel='Label',
        ylabel='Frequency',
        file_name='wine_label_distribution',
        plot=lambda: plt.hist(df['label'], bins=2, edgecolor='black')
    )
