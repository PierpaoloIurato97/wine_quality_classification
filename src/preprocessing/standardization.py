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


df = utils.read_csv('winequality-red-with-label')
standardize(df)
utils.save_csv(df, 'winequality-red-with-label-standardized')
