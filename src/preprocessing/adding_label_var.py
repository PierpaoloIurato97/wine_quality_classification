import utils


def add_label_var() -> None:
    df = utils.read_csv('winequality-red')

    if 'quality' not in df.columns:
        raise ValueError("Column 'quality' not found in raw data.")

    df['label'] = (df['quality'] > 5).astype(int)
    del df['quality']

    utils.save_csv(df, 'winequality-red-with-label')
