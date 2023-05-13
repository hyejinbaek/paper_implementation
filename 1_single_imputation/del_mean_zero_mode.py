# deletion, mean, zero, mode implemention
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
df_data = pd.read_csv(data_url)
col_data = df_data.columns = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ', 'Single Epithelial Cell Size',
                            'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
train_col = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ',
             'Single Epithelial Cell Size','Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
df_data['Bare Nuclei'] = df_data['Bare Nuclei'].replace('?',0).astype(int)
df_data['Class'] = df_data['Class'].replace({2:0, 4:1})
data = df_data[train_col].values

x = data[:,:-1]
y = data[:,-1]



def deletion_preprocessing():
    print()

def mean_preprocessing():
    print()


def zero_preprocessing():
    missing_length = 0.2
    # Create a mask for the NaN values
    nan_mask = np.random.rand(df.shape[0], df.shape[1]) < missing_length

    # Add the NaN values to the dataframe
    df[nan_mask] = np.nan
    
    df = df.copy()
    #df.loc[:missing_length-1, train_col] = np.nan
    df = df.fillna(0)

    # df_data[train_col]을 array 형태로 변경
    X = df[train_col].to_numpy()
    # df_data['Class'] : 판다스 series type을 array형태로 변경
    y = df['Class'].to_numpy()

    # X: features matrix, y: label vector
    return X, y



def mode_preprocessing():
    print()



if __name__ == '__main__':
    #deletion_preprocessing()
    #mean_preprocessing()
    zero_preprocessing()
    #mode_preprocessing()