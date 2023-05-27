# tensorflow version : 2.9.1
# mean imputation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.model_selection import train_test_split, cross_validate
from typing import Tuple

# heart_disease dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
df_data = pd.read_csv(url, header=None)

#데이터 프레임에 열 이름 추가
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


def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=32, activation='tanh', input_shape=(X.shape[1],)))
    #print(" ==== 1 model ====", model)
    model.add(tf.keras.layers.Dense(units=16, activation='tanh'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# define train code
def cross_valid(
    X: np.array, y: np.array,
    scoring=['accuracy'],
    **kwargs
):
    #model = create_model()
    model_obj = tf.keras.wrappers.scikit_learn.KerasClassifier(
        build_fn=create_model, 
        epochs=10, 
        batch_size=32
    )  # 모델 객체 생성
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model_obj.fit(x_train, y_train)  
    cv_result = cross_validate(model_obj, x_test, y_test, cv=10, scoring=scoring, **kwargs)
    print("====== cv_result ======", cv_result)
    for score_name in cv_result:
        if 'test' in score_name:
            test_score_mean, test_score_std = np.mean(cv_result[score_name]), np.std(cv_result[score_name])
            print(f'{score_name}: {test_score_mean:.4f} ± {test_score_std:.4f}')




def set_missing_value(df: pd.DataFrame) -> Tuple[np.array]:
    train_col = [
        "age",'sex', "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal"
    ]
    missing_length = 0.4

    df = df.copy()
    for col in train_col:
        nan_mask = np.random.rand(df.shape[0]) < missing_length
        df.loc[nan_mask, col] = np.nan
    print("===== df nan =====", df)
    df = df.fillna(df.mean())
    print("=====df fill na ====", df)
    X = df[train_col].to_numpy()
    y = df['target'].to_numpy()
    return X, y


if __name__ == '__main__':
    X, y = set_missing_value(df_data)
    print("missing data 20% === ")
    cross_valid(X, y)