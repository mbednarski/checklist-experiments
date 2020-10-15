import fasttext
from pprint import pprint
import scikitplot as skplt
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.trial import Trial
from sklearn.metrics import f1_score

df_test = pd.read_csv('data/processed/test.csv')
df_val = pd.read_csv('data/processed/val.csv')

def objective(trial:Trial):
    params = {
        'epoch': trial.suggest_int('epoch', 1,50),
        'dim':300,
        'minn':trial.suggest_int('minn', 1,5),
        'maxn':trial.suggest_int('maxn', 5,10),
        'wordNgrams':trial.suggest_int('wordNgrams', 1,5),
        'lr':trial.suggest_loguniform('lr', 0.001, 0.1)
    }
    model = fasttext.train_supervised(input='data/processed/train.ft', loss='softmax',verbose=0,**params,
    pretrainedVectors='wiki-news-300d-1M-subword.vec' )

    val_preds = df_val['text'].apply(lambda x: model.predict(x)[0][0][9:])
    f1 = f1_score(y_true=df_val['sentiment'], y_pred = val_preds, average='micro')

    return -f1


def train_final():
    params = {
        'epoch': 1,
        'dim':300
    }
    model = fasttext.train_supervised(input='data/processed/train.ft', loss='softmax',verbose=4,**params,
    pretrainedVectors='wiki-news-300d-1M-subword.vec' )

    val_preds = df_val['text'].apply(lambda x: model.predict(x)[0][0][9:])
    f1 = f1_score(y_true=df_val['sentiment'], y_pred = val_preds, average='micro')
    skplt.metrics.plot_confusion_matrix(df_val['sentiment'], y_pred=val_preds)
    plt.show()
    print(f1)


if __name__ == "__main__":
    train_final()


    # study_name = 'baseline-ft'
    # study = optuna.create_study(study_name=study_name, storage='sqlite:///example.db', load_if_exists=True)
    # study.optimize(objective, gc_after_trial=True, n_trials=100)

    # print(study.best_params)
    # print(study.best_value)
    