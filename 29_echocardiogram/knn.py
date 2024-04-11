import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from setproctitle import setproctitle
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.impute import KNNImputer 
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler

# CUDA 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 프로세스 제목 설정
setproctitle('hyejin')

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/res/0_final/29_echocardiogram_ensemble_method_res_final.csv'

# 결과를 저장할 리스트 초기화
results = []
rmse_list = []


class DynamicImputationModel:
    def __init__(self, num_layers, num_hidden, dim_y, num_features):
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.dim_y = dim_y
        self.num_features = num_features
        tf.compat.v1.disable_eager_execution()
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, self.num_features])

        self.y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, dim_y])
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
                # self.sess.run(self.train_op, feed_dict={self.x: batch_X.values, self.y_true: batch_y.values.reshape(-1, self.dim_y)})


    def get_accuracy(self, x_tst, y_tst):
        if self.dim_y == 1:
            pred_Y = tf.cast(self.pred > 0.5, tf.float32)
            correct_prediction = tf.equal(pred_Y, self.y_true)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            acc = self.sess.run(accuracy, feed_dict={self.x: x_tst, self.y_true: y_tst})
            # acc = self.sess.run(accuracy, feed_dict={self.x: x_tst, self.y_true: y_tst.reshape(-1, self.dim_y)})


        else:
            y_tst_hat = self.sess.run(self.pred, feed_dict={self.x: x_tst})
            y_tst_hat = np.argmax(y_tst_hat, axis=1)

            acc = accuracy_score(np.argmax(y_tst, axis=1), y_tst_hat)

        return acc
    

prepro_data = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing/29_echocardiogram.csv'
prepro_data = pd.read_csv(prepro_data)

data_pth = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/missing/29_echocardiogram.csv'
df_data = pd.read_csv(data_pth)
train_col = ['survival', 'still', 'age','pericardial', 'fractional', 'epss', 'lvdd','score','index','mult', 'group']

data_with_missing = df_data

# 반복 횟수 설정
num_iterations = 30

accuracy_list = []

for iteration in range(num_iterations):
    # Train set과 test set으로 분할
    train_data, test_data = train_test_split(data_with_missing, test_size=0.2, random_state=iteration)

    # 데이터 결측치 채우기 (KNN Imputation)
    imputer = KNNImputer(n_neighbors=1)
    train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
    test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

    # 학습을 위한 데이터 준비
    train_X = train_data.drop(columns=['class'])
    train_y = train_data['class']
    test_X = test_data.drop(columns=['class'])
    test_y = test_data['class']

    # 신경망 모델 초기화 및 학습
    model = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))
    num_epochs = 50
    batch_size = 32

    model.train_model(train_X, train_y, num_epochs, batch_size)
    accuracy = model.get_accuracy(test_X.values, test_y.values.reshape(-1, 1))
    print("==========================================")
    print(str(iteration+1)+"th accuracy === : ", accuracy)
    print("==========================================")
    accuracy_list.append(accuracy)

    model.sess.close()
    
    # 평균과 표준편차 계산
    accuracy_mean = np.mean(accuracy_list)
    accuracy_std = np.std(accuracy_list)

    # 결측치 생성 전의 데이터를 동일하게 train/test로 나누어서 저장
    original_data_train, original_data_test = train_test_split(prepro_data, test_size=0.2, random_state=iteration)
    original_data_test = original_data_test.drop(columns=['class'])

    # Min-Max Scaling 수행
    scaler = MinMaxScaler(feature_range=(-1, 1))  # imputed_test_data와 동일한 범위로 조정
    original_x_test_scaled = scaler.fit_transform(original_data_test)
    test_X_scaled = scaler.fit_transform(test_X)
    print(" == original_x_test_scaled == ", original_x_test_scaled)
    print(" == test_X_scaled == ", test_X_scaled)

    # RMSE 계산
    rmse = sqrt(mean_squared_error(original_x_test_scaled, test_X_scaled))
    print("==========================================")
    print(str(iteration + 1) + "th Ensemble Imputation rmse: ", rmse)
    print("==========================================")
    rmse_list.append(rmse)

    # 결과를 딕셔너리로 저장
    result = {
        'Dataset' : '29_echocardiogram',
        'method' : 'knn',
        'Experiment': iteration + 1,
        'RMSE': "{:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list))
    }
    results.append(result)

print("Mean Accuracy: {:.2f}".format(accuracy_mean))
print("Standard Deviation of Accuracy: {:.2f}".format(accuracy_std))
print("==========================================")
print("=== Accuracy result : {:.4f} ± {:.4f}".format(sum(accuracy_list)/len(accuracy_list), np.std(accuracy_list)))
print("=== RMSE result : {:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)))
print("==========================================")

# 결과를 DataFrame으로 변환하여 CSV 파일에 추가로 저장
results_df = pd.DataFrame(results)
if os.path.exists(result_csv_path):
    results_df.to_csv(result_csv_path, mode='a', header=False, index=False)
else:
    results_df.to_csv(result_csv_path, index=False)

print("Results saved to:", result_csv_path)
