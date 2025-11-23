import utils

COLUMNS_TO_STANDARDIZE = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]


def standardize():
    df = utils.read_csv('winequality-red-with-label')

    for col in COLUMNS_TO_STANDARDIZE:
        utils.ensure_col_exists(df, col)

    df[COLUMNS_TO_STANDARDIZE] = (
        df[COLUMNS_TO_STANDARDIZE] - df[COLUMNS_TO_STANDARDIZE].mean()
    ) / df[COLUMNS_TO_STANDARDIZE].std()

    utils.save_csv(df, 'winequality-red-with-label-standardized')
