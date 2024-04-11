# missing value 20% 생성한 뒤 결측치 있는 데이터셋 따로 저장
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior

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

result_preprocessing_path = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing/33_forty.csv'
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/missing/33_forty.csv'

# 데이터 파일 경로 설정
data_pth = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/original/33_forty.csv'

# 데이터 불러오기
df_data = pd.read_csv(data_pth)

# df_data.columns = ['class', 'Repetition', 'PH', 'IFP', 'NLP', 'NGP', 'NGL', 'NS', 'MHG', 'GY']
# df_data = df_data.drop(labels='id', axis=1)

# df_data = df_data.replace(1, 0).astype(str)
# train_col = ['Repetition', 'PH', 'IFP', 'NLP', 'NGP', 'NGL', 'NS', 'MHG', 'GY']


# unique_values_A2 = df_data['liver_big'].unique()
# print("Unique values in column A2:", unique_values_A2)
# train_col = ['age', 'oper', 'positive']


# col_data = df_data.columns
# train_col = list(col_data)
# train_col.remove('class')

data = df_data


# categorical_columns = ['class']

# # 레이블 인코딩 적용
# df_encoded = label_encode(df_data, categorical_columns)
# data = df_encoded

# # df_data = pd.get_dummies(df_data, columns=['class'], prefix='class')
# print(data)

# # 데이터전처리 후 데이터프레임을 CSV 파일로 저장
# data.to_csv(result_preprocessing_path, index=False)
# print("Data preprocessing saved to CSV file.")

# missing_length = 0.2
# for col in train_col:
#     nan_mask = np.random.rand(data.shape[0]) < missing_length
#     data.loc[nan_mask, col] = np.nan

# data_with_missing = data
# print(data_with_missing)

# # 결측치 생성 후 데이터프레임을 CSV 파일로 저장
# data_with_missing.to_csv(result_csv_path, index=False)
# print("Data with missing values saved to CSV file.")