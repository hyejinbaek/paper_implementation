import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from setproctitle import setproctitle
from tensorflow.keras.layers import Input, Embedding, Flatten
from sklearn.preprocessing import LabelEncoder
from datawig import SimpleImputer
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.metrics import accuracy_score

# CUDA 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 프로세스 제목 설정
setproctitle('hyejin')

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/experiment_result.csv'

# 결과를 저장할 리스트 초기화
results = []

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
data_pth = './iris.data'

# 데이터 불러오기
df_data = pd.read_csv(data_pth)
print(df_data)
col_data = df_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
train_col = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

data = df_data

# 범주형 피처 선택
categorical_columns = ['class']

# 레이블 인코딩 적용
df_encoded = label_encode(df_data, categorical_columns)
data = df_encoded

missing_length = 0.2
for col in train_col:
    nan_mask = np.random.rand(data.shape[0]) < missing_length
    data.loc[nan_mask, col] = np.nan

data_with_missing = data
print(data_with_missing)

# # 반복 횟수 설정
# num_iterations = 10

# accuracy_list = []
# imputers = {}

# for iteration in range(num_iterations):
#     # Train set과 test set으로 분할
#     train_data, test_data = train_test_split(data_with_missing, test_size=0.2, random_state=iteration)

#     # 데이터 결측치 채우기
#     for col in train_col:
#         imputer = SimpleImputer(
#             input_columns=train_col,
#             output_column=col,
#             output_path=f'./imputer_model/imputer_model_{col}'
#         )
#         imputer.fit(train_df=train_data, num_epochs=5)
#         imputers[col] = imputer

#     # Impute missing values for each column in train_data
#     train_imputed_data = {}
#     for col, imputer in imputers.items():
#         predictions = imputer.predict(train_data)
#         train_imputed_data[col] = predictions[col + '_imputed']  # '_imputed' is added by datawig

#     # Create a DataFrame with imputed values for train set
#     train_imputed_df = pd.DataFrame(train_imputed_data)

#     # Impute missing values for each column in test_data
#     test_imputed_data = {}
#     for col, imputer in imputers.items():
#         predictions = imputer.predict(test_data)
#         test_imputed_data[col] = predictions[col + '_imputed']  # '_imputed' is added by datawig

#     # Create a DataFrame with imputed values for test set
#     test_imputed_df = pd.DataFrame(test_imputed_data)

#     # 학습을 위한 데이터 준비
#     train_X = train_imputed_df[train_col].values  # Select only the columns for training
#     train_y = train_data['class'].values  # Convert to NumPy array
#     test_X = test_imputed_df[train_col].values  # Select only the columns for testing
#     test_y = test_data['class'].values  # Convert to NumPy array

#     # 신경망 모델 학습
#     model = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))  # Pass num_features
#     num_epochs = 50
#     batch_size = 32
#     model.train_model(train_X, train_y, num_epochs, batch_size)

#     accuracy = model.get_accuracy(test_X, test_y.reshape(-1, 1))
#     print("==========================================")
#     print(str(iteration+1)+"th accuracy === : ", accuracy)
#     print("==========================================")
#     accuracy_list.append(accuracy)

#     model.sess.close()

#     # 모든 반복이 끝난 후에 평균 및 표준편차 계산
#     accuracy_mean = np.mean(accuracy_list)
#     accuracy_std = np.std(accuracy_list)

#     # 결과를 딕셔너리로 저장
#     result = {
#         'Dataset' : '7_post_patient',
#         'method' : 'datawig',
#         'Experiment': iteration + 1,
#         'Accuracy': "{:.4f} ± {:.4f}".format(accuracy, np.std(accuracy))
#     }
#     results.append(result)

# print("Mean Accuracy: {:.2f}".format(accuracy_mean))
# print("Standard Deviation of Accuracy: {:.2f}".format(accuracy_std))
# print("==========================================")
# print("=== result : {:.4f} ± {:.4f}".format(sum(accuracy_list)/len(accuracy_list), np.std(accuracy_list)))
# print("==========================================")

# # 결과를 DataFrame으로 변환하여 CSV 파일에 추가로 저장
# results_df = pd.DataFrame(results)
# if os.path.exists(result_csv_path):
#     results_df.to_csv(result_csv_path, mode='a', header=False, index=False)
# else:
#     results_df.to_csv(result_csv_path, index=False)

# print("Results saved to:", result_csv_path)
