import pandas as pd
import utils

COLUMNS_TO_STANDARDIZE = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]


def standardize(df: pd.DataFrame):
    df[COLUMNS_TO_STANDARDIZE] = (
        df[COLUMNS_TO_STANDARDIZE] - df[COLUMNS_TO_STANDARDIZE].mean()
    ) / df[COLUMNS_TO_STANDARDIZE].std()


df = utils.read_csv('winequality-red-processed')
standardize(df)
utils.print_descriptive_statistics(df)
utils.save_csv(df, 'winequality-red-processed-standardized')
