import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv('data/processed/full_dset.csv')

    df_keep, df_test = train_test_split(df, test_size=0.2, stratify=df['sentiment'], random_state=985)
    df_train, df_val = train_test_split(df_keep, test_size=0.2, stratify=df_keep['sentiment'], random_state=985)

    print('Train size:', len(df_train))
    print('Val size:', len(df_val))
    print('Test size:', len(df_test))

    df_train.to_csv('data/processed/train.csv', index=False)
    df_val.to_csv('data/processed/val.csv', index=False)
    df_test.to_csv('data/processed/test.csv', index=False)

    