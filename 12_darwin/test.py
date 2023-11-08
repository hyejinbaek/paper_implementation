import os
# CUDA 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from setproctitle import setproctitle
from tensorflow.keras.layers import Input, Embedding, Flatten
from sklearn.preprocessing import LabelEncoder

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/res/1_breast_ensemble_method_res.csv'

# 결과를 저장할 리스트 초기화
results = []



# 프로세스 제목 설정
setproctitle('hyejin')

def label_encode(df, columns):
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for col in columns:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col].astype(str))
    return df_encoded

def build_embedding_model(input_dims, embedding_dims):
    inputs = []
    embeddings = []
    for input_dim in input_dims:
        input_layer = Input(shape=(1,))
        embedding = Embedding(input_dim, embedding_dims)(input_layer)
        embedding = Flatten()(embedding)
        inputs.append(input_layer)
        embeddings.append(embedding)
    return inputs, embeddings

data_pth = './darwin.csv'

# 데이터 불러오기
df_data = pd.read_csv(data_pth, usecols=lambda column: column != 'ID')
col_data = df_data.columns
train_col = list(col_data)
train_col.remove('class')
data = df_data

categorical_columns = ['class']
# 레이블 인코딩 적용
df_encoded = label_encode(df_data, categorical_columns)
data = df_encoded

missing_length = 0.2
for col in train_col:
    nan_mask = np.random.rand(data.shape[0]) < missing_length
    data.loc[nan_mask, col] = np.nan

data_with_missing = data



