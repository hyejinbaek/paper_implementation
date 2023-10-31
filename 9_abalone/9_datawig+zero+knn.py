import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from setproctitle import setproctitle
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior
from sklearn.metrics import accuracy_score
from datawig import SimpleImputer
from sklearn.impute import KNNImputer
from math import sqrt


# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/res/9_abalone_ensemble_method_res.csv'

# 결과를 저장할 리스트 초기화
results = []

# CUDA 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 프로세스 제목 설정
setproctitle('hyejin')

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
data_pth = './abalone.data'

# 데이터 불러오기
df_data = pd.read_csv(data_pth)
col_data = df_data.columns = ['class','Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
train_col =  ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
df_data['class'] = df_data['class'].replace({'M':0, 'F':1, 'I':2})
data = df_data

missing_length = 0.2
for col in train_col:
    nan_mask = np.random.rand(data.shape[0]) < missing_length
    data.loc[nan_mask, col] = np.nan

data_with_missing = data

# 반복 횟수 설정
num_iterations = 10

accuracy_list = []
imputers = {}

for iteration in range(num_iterations):
    # Train set과 test set으로 분할
    train_data, test_data = train_test_split(data_with_missing, test_size=0.2, random_state=iteration)

    ## datawig
    for col in train_col:
        imputer = SimpleImputer(
            input_columns=train_col,
            output_column=col,
            output_path=f'./imputer_model/imputer_model_{col}'
        )
        imputer.fit(train_df=train_data, num_epochs=5)
        imputers[col] = imputer

    # Impute missing values for each column in train_data
    train_imputed_data = {}
    for col, imputer in imputers.items():
        predictions = imputer.predict(train_data)
        train_imputed_data[col] = predictions[col + '_imputed']

    # Create a DataFrame with imputed values for train set
    train_imputed_df = pd.DataFrame(train_imputed_data)

    # Impute missing values for each column in test_data
    test_imputed_data = {}
    for col, imputer in imputers.items():
        predictions = imputer.predict(test_data)
        test_imputed_data[col] = predictions[col + '_imputed']

    # Create a DataFrame with imputed values for test set
    test_imputed_df = pd.DataFrame(test_imputed_data)

    # 학습을 위한 데이터 준비
    train_X = train_imputed_df[train_col]
    train_y = train_data['class']
    test_X = test_imputed_df[train_col]
    test_y = test_data['class']

    # 데이터 결측치 채우기 (Zero Imputation)
    train_data_zero_imputed = train_data.fillna(0)
    test_data_zero_imputed = test_data.fillna(0)

    # zero imputation 데이터 준비
    train_X_zero_imputed = train_data_zero_imputed.drop(columns=['class'])
    train_y_zero_imputed = train_data_zero_imputed['class']
    test_X_zero_imputed = test_data_zero_imputed.drop(columns=['class'])
    test_y_zero_imputed = test_data_zero_imputed['class']

     # 데이터 결측치 채우기 (KNN Imputation)
    imputer = KNNImputer(n_neighbors=5)
    train_data_knn_imputed = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
    test_data_knn_imputed = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

    # 학습을 위한 데이터 준비
    train_X_knn_imputed = train_data_knn_imputed.drop(columns=['class'])
    train_y_knn_imputed = train_data_knn_imputed['class']
    test_X_knn_imputed = test_data_knn_imputed.drop(columns=['class'])
    test_y_knn_imputed = test_data_knn_imputed['class']

    # datawig 신경망 모델 학습
    model_datawig_imputation = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))
    model_datawig_imputation.train_model(train_X, train_y, num_epochs=50, batch_size=32)
    accuracy = model_datawig_imputation.get_accuracy(test_X.values, test_y.values.reshape(-1, 1))

    # 신경망 모델 초기화 및 학습 (Zero Imputation)
    model_zero_imputation = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))
    model_zero_imputation.train_model(train_X_zero_imputed, train_y_zero_imputed, num_epochs=50, batch_size=32)
    accuracy_zero_imputation = model_zero_imputation.get_accuracy(test_X_zero_imputed.values, test_y_zero_imputed.values.reshape(-1, 1))


    # 신경망 모델 초기화 및 학습 (KNN Imputation)
    model_knn_imputation = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))
    model_knn_imputation.train_model(train_X_knn_imputed, train_y_knn_imputed, num_epochs=50, batch_size=32)
    accuracy_knn_imputation = model_knn_imputation.get_accuracy(test_X_knn_imputed.values, test_y_knn_imputed.values.reshape(-1, 1))
    
    print("==========================================")
    print(str(iteration + 1) + "th Datawig Imputation accuracy: ", accuracy)
    print(str(iteration + 1) + "th knn Imputation accuracy: ", accuracy_knn_imputation)
    print(str(iteration + 1) + "th Zero Imputation accuracy: ", accuracy_zero_imputation)
    print("==========================================")

    accuracy_list.append(accuracy)
    accuracy_list.append(accuracy_knn_imputation)
    accuracy_list.append(accuracy_zero_imputation)

    # 결과를 딕셔너리로 저장 (Ensemble 결과)
    result = {
        'Dataset': '9_abalone',
        'method': '9_datawig+zero+knn.py',
        'Experiment': iteration + 1,
        'Accuracy': "{:.4f} ± {:.4f}".format(np.mean(accuracy_list), np.std(accuracy_list))
    }
    results.append(result)

print("Mean Ensemble Accuracy: {:.4f}".format(np.mean(accuracy_list)))
print("Standard Deviation of Ensemble Accuracy: {:.4f}".format(np.std(accuracy_list)))
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
