# tensorflow version: 2.9.1
# mean imputation
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.model_selection import train_test_split
from typing import Tuple
from sklearn.metrics import accuracy_score

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
df_data = pd.read_csv(data_url)
col_data = df_data.columns = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ', 'Single Epithelial Cell Size',
                            'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
train_col = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ',
             'Single Epithelial Cell Size','Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
df_data['Bare Nuclei'] = df_data['Bare Nuclei'].replace('?',0).astype(int)
df_data['Class'] = df_data['Class'].replace({2:0, 4:1})

def create_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=32, activation='tanh', input_shape=input_shape))
    print(" ==== 1 model ====", model)
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
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
        y_pred = (model.predict(x_test) > 0.5).astype(int).flatten()
        acc = accuracy_score(y_test, y_pred)
        print(str(i+1)+"th accuracy === : ", acc)
        acc_list.append(acc)
    print("mean acc : {}".format(sum(acc_list)/len(acc_list)))
    print("std acc : {}".format(np.std(acc_list)))

def set_missing_value(df: pd.DataFrame) -> Tuple[np.array]:
    train_col = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ',
             'Single Epithelial Cell Size','Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    missing_length = 0.2

    df = df.copy()
    print(" ==== 1 df =====", df)
    for col in train_col:
        nan_mask = np.random.rand(df.shape[0]) < missing_length
        df.loc[nan_mask, col] = np.nan
    print(" ==== 2 df =====", df)

    # df[train_col].mode().iloc[0]은 각 열의 최빈값을 구한 뒤, 첫 번째 값을 사용하여 결측값을 대체
    df[train_col] = df[train_col].fillna(df[train_col].mode().iloc[0])
    print(" ==== 3 df fill mode =====", df)
    X = df[train_col].to_numpy()
    y = df['Class'].to_numpy()
    return X, y


if __name__ == '__main__':
    X, y = set_missing_value(df_data)
    print("missing data 20% === ")
    cross_valid(X, y)
