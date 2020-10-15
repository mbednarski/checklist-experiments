import transformers
import pandas as pd
from pprint import pprint
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import scikitplot as skplt
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df_train = pd.read_csv('data/processed/train.csv')
    df_test = df_test[df_test['sentiment']!='neutral']
    dset = df_test['text'].tolist()

    y_trues = df_test['sentiment'].str.upper()

    preds = []
    loader = DataLoader(dset, batch_size=64, shuffle=False)
    pipe = transformers.pipeline('sentiment-analysis', device=0)
    for batch in tqdm(loader):
        batch_preds = pipe.predict(batch)
        preds += batch_preds

    y_pred = [x['label'] for x in preds]
    f1_score(y_true=y_trues, y_pred =y_pred, average='micro')
    skplt.metrics.plot_confusion_matrix(y_true=y_trues, y_pred=y_pred)
    plt.show()