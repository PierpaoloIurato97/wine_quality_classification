import matplotlib.pyplot as plt

import utils


def describe_and_plot_processed_data():
    df = utils.read_csv('winequality-red-with-label-standardized')

    if 'label' not in df.columns:
        raise ValueError("Column 'label' not found in processed data.")

    print("Descriptive statistics for processed data:", '\n')
    utils.print_descriptive_statistics(df)

    print("Class distribution in processed data:")
    print(df['label'].value_counts())

    utils.make_plot(
        title='Processed Wine Label Distribution',
        xlabel='Label',
        ylabel='Frequency',
        file_name='wine_label_distribution',
        plot=lambda: plt.hist(df['label'], bins=2, edgecolor='black')
    )
