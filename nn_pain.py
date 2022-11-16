!pip install --pre pycaret

from google.colab import files
arquivo = files.upload()

import pandas as pd

df = pd.read_excel('withlabels3.xlsx')

df.head()

df.describe()

df.info()

df.describe().transpose()

df2 = pd.get_dummies(df, prefix=None, prefix_sep='_', dummy_na=False, columns=['typeofsurg', 'sint2', 'etiol','side','complain', 'modic'], sparse=False, drop_first=False, dtype=int)

df2.info()

def index_of_dic(dic, key):
    return dic[key]

def StrList_to_UniqueIndexList(lista):
    group = set(lista)
    
    dic = {}
    i = 0
    for g in group:
        if g not in dic:
            dic[g] = i
            i += 1

    return [index_of_dic(dic, p) for p in lista]


df2['priorsurg'] = StrList_to_UniqueIndexList(df2['priorsurg'])
df2['insurance'] = StrList_to_UniqueIndexList(df2['insurance'])
df2['autonomy'] = StrList_to_UniqueIndexList(df2['autonomy'])
df2['hiz'] = StrList_to_UniqueIndexList(df2['hiz'])
df2['motorpre'] = StrList_to_UniqueIndexList(df2['motorpre'])
df2['senspre'] = StrList_to_UniqueIndexList(df2['senspre'])
df2['reflpre'] = StrList_to_UniqueIndexList(df2['reflpre'])
df2['work'] = StrList_to_UniqueIndexList(df2['work'])
df2['sex'] = StrList_to_UniqueIndexList(df2['sex'])
#clientes['name'] = StrList_to_UniqueIndexList(clientes['Category'])

df2.info()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df2), columns = df2.columns)
df.head()

df2.corr()

# importar os pacotes necessários
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=10)
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

df.loc[df['cmid'] == 'yes', 'cmid'] = 1
df.loc[df['cmid'] == 'no', 'cmid'] = 0

X = df.drop('cmid',1)
y = df['cmid']

y = y.astype('int64')

from imblearn.over_sampling import SMOTE


X_resampled, y_resampled = SMOTE().fit_resample(X, y)

X_resampled.shape, y_resampled.shape

X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
X_train.shape, y_train.shape

X = X_train

y = y_train

X_resampled.shape, y_resampled.shape

from pycaret.classification import *
s = setup(X, target = y, preprocess=False)

best = compare_models(sort='auc')

print(best)

models()

mlp = create_model('mlp')

tuned_mlp = tune_model(mlp, optimize='auc')

!pip install matplotlib==3.1.1
import matplotlib.pyplot as plt

pip install seaborn

# plotando matriz de confusão
plot_model(tuned_mlp, plot='confusion_matrix')

# avaliando o modelo
evaluate_model(tuned_mlp)

predict_model(tuned_mlp)

final_mlp = finalize_model(tuned_mlp)

print(final_mlp)

calibrated_mlp = calibrate_model(mlp,method='isotonic', return_train_score=True)

plot_model(mlp, plot='confusion_matrix')

plot_model(mlp, plot='confusion_matrix', use_train_data=False, plot_kwargs={'percent': True})

plot_model(calibrated_mlp, plot='confusion_matrix', use_train_data=False, plot_kwargs={'percent': True})

plot_model(calibrated_mlp, plot='calibration')

plot_model(mlp, plot = 'auc', use_train_data = True)

plot_model(tuned_mlp, plot = 'auc')

plot_model(calibrated_mlp, plot = 'auc')

save_model(tuned_mlp,"vas1y")

from sklearn.metrics import brier_score_loss
add_metric('brierscoreloss','Brier Score Loss', brier_score_loss, greater_is_better=False)

get_metrics()