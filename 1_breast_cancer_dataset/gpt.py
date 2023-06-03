import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from setproctitle import *
setproctitle('hyejin')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# 데이터셋 로드
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(data_url, header=None, names=column_names, na_values='?')

# 결측치 생성
df_with_nan = df.copy()
nan_ratio = 0.2
nan_mask = np.random.rand(*df_with_nan.shape) < nan_ratio
df_with_nan[nan_mask] = np.nan

# 결측치 처리 및 데이터셋 분할
X = df_with_nan.drop('target', axis=1)
y = df_with_nan['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train 세트 Mean Imputation
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Test 세트 Mean Imputation
X_test_imputed = imputer.transform(X_test)

# Label Encoding
label_encoder = LabelEncoder()
X_train_encoded = X_train_imputed.copy()
X_test_encoded = X_test_imputed.copy()
for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
    X_train_encoded[col] = label_encoder.fit_transform(X_train_encoded[col].astype(str))
    X_test_encoded[col] = label_encoder.transform(X_test_encoded[col].astype(str))

# 모델 정의
embedding_dims = 8
inputs = []
embeddings = []
for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
    input_dim = len(df[col].unique())
    input_layer = Input(shape=(1,))
    embedding = Embedding(input_dim, embedding_dims)(input_layer)
    embedding = Flatten()(embedding)
    inputs.append(input_layer)
    embeddings.append(embedding)

# 숫자형 피처
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
numeric_input = Input(shape=(len(numeric_features),))
inputs.append(numeric_input)

# Concatenate
x = Concatenate()(embeddings + [numeric_input])

# 모델 정의
hidden_dims = 32
x = Dense(hidden_dims, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

# 모델 생성
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit([X_train_encoded[col].values for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']] + [X_train_imputed[numeric_features].values],
          y_train.values,
          epochs=10,
          batch_size=32,
          verbose=1)

# Test 세트 평가
y_pred = model.predict([X_test_encoded[col].values for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']] + [X_test_imputed[numeric_features].values])
y_pred = y_pred > 0.5
accuracy = (y_pred == y_test.values.reshape(-1, 1)).mean()
print("Accuracy:", accuracy)
