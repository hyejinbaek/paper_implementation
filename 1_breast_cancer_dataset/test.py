import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from setproctitle import *
setproctitle('hyejin')
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.model_selection import train_test_split
from typing import Tuple
from sklearn.metrics import accuracy_score

def get_dataset():
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    df_data = pd.read_csv(data_url)
    col_data = df_data.columns = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ', 'Single Epithelial Cell Size',
                                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    df_data['Bare Nuclei'] = df_data['Bare Nuclei'].replace('?',0).astype(int)
    df_data['Class'] = df_data['Class'].replace({2:0, 4:1})
    df = df_data
    return df

def create_mv(df):
    missing_length = 0.2
    train_col = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ',
                'Single Epithelial Cell Size','Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    df = df.copy()
    for col in train_col:
        nan_mask = np.random.rand(df.shape[0]) < missing_length
        df.loc[nan_mask, col] = np.nan

    return df

def create_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=32, activation='tanh', input_shape=input_shape))
    print(" ==== 1 model ====", model)
    model.add(tf.keras.layers.Dense(units=16, activation='tanh'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

