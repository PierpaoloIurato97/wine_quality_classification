import pandas as pd
import utils


def add_label_variable(df: pd.DataFrame) -> None:
    if 'quality' not in df.columns:
        raise ValueError("Column 'quality' not found in raw data.")

    df['label'] = (df['quality'] > 5).astype(int)
    del df['quality']


df = utils.read_csv('winequality-red')
add_label_variable(df)
utils.save_csv(df, 'winequality-red-processed')
