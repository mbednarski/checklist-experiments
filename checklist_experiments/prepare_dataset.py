import pandas as pd
from sklearn.model_selection import train_test_split
import re

def preprocess(s:str):
    s = s.lower()
    s = re.sub('[^a-zA-Z0-9@]', ' ', s)
    s = re.sub('\s+', ' ', s)
    return s
    

if __name__ == "__main__":
    df = pd.read_csv("data/processed/full_dset.csv")

    df['text'] = df['text'].apply(preprocess)

    df_keep, df_test = train_test_split(
        df, test_size=0.2, stratify=df["sentiment"], random_state=985
    )
    df_train, df_val = train_test_split(
        df_keep, test_size=0.2, stratify=df_keep["sentiment"], random_state=985
    )

    print("Train size:", len(df_train))
    print("Val size:", len(df_val))
    print("Test size:", len(df_test))

    ft_train= df_train['text'] + '__label__' + df_train['sentiment']
    ft_train.to_csv('data/processed/train.ft', index=False, header=None)

    ft_val= df_val['text'] + '__label__' + df_val['sentiment']
    ft_val.to_csv('data/processed/val.ft', index=False, header=None)

    ft_test= df_test['text'] + '__label__' + df_test['sentiment']
    ft_test.to_csv('data/processed/test.ft', index=False, header=None)

    df_train.to_csv("data/processed/train.csv", index=False)
    df_val.to_csv("data/processed/val.csv", index=False)
    df_test.to_csv("data/processed/test.csv", index=False)
