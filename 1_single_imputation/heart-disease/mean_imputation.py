# tensorflow version : 2.9.1
# mean imputation
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

# heart_disease dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
df_data = pd.read_csv(url, header=None)

#데이터 프레임에 열 이름 추가
# sex, fbs, exang
column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
df_data.columns = column_names
df_data['ca'] = df_data['ca'].replace('?',0.0).astype(str)
df_data['thal'] = df_data['thal'].replace('?',0.0).astype(str)
df_data['ca']= df_data['ca'].astype(dtype='float64')
df_data['thal']= df_data['thal'].astype(dtype='float64')
print(df_data)


def create_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=32, activation='tanh', input_shape=input_shape))
    model.add(tf.keras.layers.Dense(units=16, activation='tanh'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# define train code
def cross_valid(X: np.array, y: np.array):
    acc_list = []
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=i)
        model = create_model((X.shape[1],))
        model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
        y_pred = (model.predict(x_test) > 0.5).astype(int).flatten()
        acc = accuracy_score(y_test, y_pred)
        print(str(i+1)+"th accuracy === : ", acc)
        acc_list.append(acc)
    print("mean acc : {}".format(sum(acc_list)/len(acc_list)))
    print("std acc : {}".format(np.std(acc_list)))


def set_missing_value(df: pd.DataFrame) -> Tuple[np.array]:
    train_col = [
        "age", "cp", "trestbps", "chol",  "restecg", "thalach",
        "oldpeak", "slope", "ca", "thal"
    ]
    missing_length = 0.2

    df = df.copy()
    for col in train_col:
        nan_mask = np.random.rand(df.shape[0]) < missing_length
        df.loc[nan_mask, col] = np.nan

    df = df.fillna(df.mean())
    X = df[train_col].to_numpy()
    y = df['target'].to_numpy()
    return X, y

if __name__ == '__main__':
    X, y = set_missing_value(df_data)
    print("missing data 20% === ")
    cross_valid(X, y)
