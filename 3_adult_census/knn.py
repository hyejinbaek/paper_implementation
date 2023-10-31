import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from setproctitle import setproctitle
from tensorflow.keras.layers import Input, Embedding, Flatten
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.impute import KNNImputer 
from sklearn.metrics import accuracy_score


# CUDA 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

class DynamicImputationModel:
    def __init__(self, num_layers, num_hidden, dim_y):
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.dim_y = dim_y
        tf.compat.v1.disable_eager_execution()
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, train_X.shape[1]])
        self.y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.logits, self.pred = self.build_model(self.x)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_true, logits=self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = self.optimizer.minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, x):
        for _ in range(self.num_layers):
            x = tf.layers.dense(x, self.num_hidden, activation=tf.nn.tanh)
        logits = tf.layers.dense(x, self.dim_y)

        if self.dim_y == 1:
            pred = tf.nn.sigmoid(logits)
        elif self.dim_y > 2:
            pred = tf.nn.softmax(logits)

        return logits, pred

    def train_model(self, train_X, train_y, num_epochs, batch_size):
        num_batches = int(np.ceil(len(train_X) / batch_size))
        for epoch in range(num_epochs):
            indices = np.arange(len(train_X))
            np.random.shuffle(indices)
            train_X_shuffled = train_X.iloc[indices]
            train_y_shuffled = train_y.iloc[indices]

            for i in range(num_batches):
                batch_X = train_X_shuffled.iloc[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y_shuffled.iloc[i * batch_size: (i + 1) * batch_size]

                self.sess.run(self.train_op, feed_dict={self.x: batch_X.values, self.y_true: batch_y.values.reshape(-1, 1)})

    def get_accuracy(self, x_tst, y_tst):
        if self.dim_y == 1:
            pred_Y = tf.cast(self.pred > 0.5, tf.float32)
            correct_prediction = tf.equal(pred_Y, self.y_true)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            acc = self.sess.run(accuracy, feed_dict={self.x: x_tst, self.y_true: y_tst})

        else:
            y_tst_hat = self.sess.run(self.pred, feed_dict={self.x: x_tst})
            y_tst_hat = np.argmax(y_tst_hat, axis=1)

            acc = accuracy_score(np.argmax(y_tst, axis=1), y_tst_hat)

        return acc

# 데이터 파일 경로 설정
data_pth = './adult.data'

# 데이터 불러오기
df_data = pd.read_csv(data_pth)
df_data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status','occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'target']
train_col = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status','occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country']
df_data['workclass'] = df_data['workclass'].replace('?', 0).astype(str)
df_data['occupation'] = df_data['occupation'].replace('?', 0).astype(str)
data = df_data

# 범주형 피처 선택
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation','relationship', 'race', 'sex', 'native-country', 'target']

# 레이블 인코딩 적용
df_encoded = label_encode(df_data, categorical_columns)
data = df_encoded

missing_length = 0.2
for col in train_col:
    nan_mask = np.random.rand(data.shape[0]) < missing_length
    data.loc[nan_mask, col] = np.nan

data_with_missing = data

# KNNImputer를 사용하여 결측치 처리
imputer = KNNImputer(n_neighbors=5)  # 이웃 개수 조정 가능
data_imputed = imputer.fit_transform(data_with_missing)

# 반복 횟수 설정
num_iterations = 10

accuracy_list = []
rmse_list = []

for iteration in range(num_iterations):
    # Train set과 test set으로 분할
    train_data, test_data = train_test_split(data_with_missing, test_size=0.2, random_state=iteration)

    # 데이터 결측치 채우기 (KNN Imputation)
    imputer = KNNImputer(n_neighbors=5)
    train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
    test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

    # 학습을 위한 데이터 준비
    train_X = train_data.drop(columns=['target'])
    train_y = train_data['target']
    test_X = test_data.drop(columns=['target'])
    test_y = test_data['target']

    # 신경망 모델 초기화 및 학습
    model = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1)
    num_epochs = 50
    batch_size = 32

    model.train_model(train_X, train_y, num_epochs, batch_size)
    accuracy = model.get_accuracy(test_X.values, test_y.values.reshape(-1, 1))
    print("==========================================")
    print(str(iteration+1)+"th accuracy === : ", accuracy)
    print("==========================================")
    accuracy_list.append(accuracy)
    
    # 평균과 표준편차 계산
    accuracy_mean = np.mean(accuracy_list)
    accuracy_std = np.std(accuracy_list)

    # 결과를 딕셔너리로 저장
    result = {
        'Dataset' : '3_adult',
        'method' : 'knn',
        'Experiment': iteration + 1,
        'Accuracy': "{:.4f} ± {:.4f}".format(accuracy, np.std(accuracy))
    }
    results.append(result)

print("Mean Accuracy: {:.2f}".format(accuracy_mean))
print("Standard Deviation of Accuracy: {:.2f}".format(accuracy_std))
print("==========================================")
print("=== result : {:.4f} ± {:.4f}".format(sum(accuracy_list)/len(accuracy_list), np.std(accuracy_list)))
print("==========================================")


# 결과를 DataFrame으로 변환하여 CSV 파일에 추가로 저장
results_df = pd.DataFrame(results)
if os.path.exists(result_csv_path):
    results_df.to_csv(result_csv_path, mode='a', header=False, index=False)
else:
    results_df.to_csv(result_csv_path, index=False)

print("Results saved to:", result_csv_path)
