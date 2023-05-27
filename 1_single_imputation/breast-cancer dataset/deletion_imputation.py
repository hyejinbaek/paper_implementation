# tensorflow version : 2.9.1
# deletion imputation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.model_selection import train_test_split, cross_validate, KFold
from typing import Tuple


data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
df_data = pd.read_csv(data_url)
col_data = df_data.columns = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ', 'Single Epithelial Cell Size',
                            'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
train_col = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ',
             'Single Epithelial Cell Size','Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
df_data['Bare Nuclei'] = df_data['Bare Nuclei'].replace('?',0).astype(int)
df_data['Class'] = df_data['Class'].replace({2:0, 4:1})

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=32, activation='tanh', input_shape=(X.shape[1],)))
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
    model_obj = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=10, batch_size=32)  # 모델 객체 생성
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model_obj.fit(x_train, y_train)
    cv_result = cross_validate(model_obj, x_test, y_test, cv=10, scoring=scoring, **kwargs)
    print("====== cv_result ======", cv_result)
    for score_name in cv_result:
        if 'test' in score_name:
            test_score_mean, test_score_std = np.mean(cv_result[score_name]), np.std(cv_result[score_name])
            print(f'{score_name}: {test_score_mean:.4f} ± {test_score_std:.4f}')
            
def set_missing_value(df: pd.DataFrame) -> Tuple[np.array]:

    missing_length = 0.2

    # Create a mask for the NaN values
    nan_mask = np.random.rand(df.shape[0], df.shape[1]) < missing_length

    # Add the NaN values to the dataframe
    df[nan_mask] = np.nan

    df = df.copy()
    print("==== df nan ====", df)
    df = df.dropna()
    print("==== df drop ====", df)
    X = df[train_col].to_numpy()
    y = df['Class'].to_numpy()
    # X: features matrix, y: label vector
    return X, y

if __name__ == '__main__':
    X, y = set_missing_value(df_data)
    print("missing data 20% === ")
    cross_valid(X, y)
