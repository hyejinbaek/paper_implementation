import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Embedding, Flatten
from sklearn.preprocessing import LabelEncoder

# 16_rice_ensemble_method_res

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

# 데이터 파일 경로 설정
data_pth = './Rice_Cammeo_Osmancik.arff'
df_data = pd.read_csv(data_pth)
print(df_data)
# 데이터 불러오기
df_data = pd.read_csv(data_pth)
df_data.columns = ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 'Eccentricity', 'Convex_Area', 'Extent', 'class']
train_col = ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 'Eccentricity', 'Convex_Area', 'Extent']

categorical_columns = ['class']
# 레이블 인코딩 적용
df_encoded = label_encode(df_data, categorical_columns)
data = df_encoded
print(data)