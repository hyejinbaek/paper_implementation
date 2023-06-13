import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from setproctitle import *
setproctitle('hyejin')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datawig
from sklearn.metrics import accuracy_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class DynamicImputationModel:
    def __init__(self, num_layers, num_hidden, dim_y, train_X):
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.dim_y = dim_y
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, train_X.shape[1]])
        self.y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, dim_y])
        self.logits, self.pred = self.build_model(self.x)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true, logits=self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = self.optimizer.minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, x):
        for _ in range(self.num_layers):
            x = tf.layers.dense(x, self.num_hidden, activation=tf.nn.tanh)
        logits = tf.layers.dense(x, self.dim_y)
        pred = tf.nn.softmax(logits)
        return logits, pred

    def train_model(self, train_X, train_y, num_epochs, batch_size):
        num_batches = int(np.ceil(len(train_X) / batch_size))
        for epoch in range(num_epochs):
            indices = np.arange(len(train_X))
            np.random.seed(epoch)
            np.random.shuffle(indices)
            train_X_shuffled = train_X.iloc[indices]
            train_y_shuffled = train_y.iloc[indices]

            for i in range(num_batches):
                batch_X = train_X_shuffled.iloc[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y_shuffled.iloc[i * batch_size: (i + 1) * batch_size]

                self.sess.run(self.train_op, feed_dict={self.x: batch_X.values, self.y_true: batch_y.values})

    def get_accuracy(self, x_tst, y_tst):
        y_tst_hat = self.sess.run(self.pred, feed_dict={self.x: x_tst})
        y_tst_hat = np.argmax(y_tst_hat, axis=1)
        acc = accuracy_score(np.argmax(y_tst, axis=1), y_tst_hat)
        return acc

data_pth = './abalone.data'

df_data = pd.read_csv(data_pth)
col_data = df_data.columns = ['class','Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
train_col =  ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
df_data['class'] = df_data['class'].replace({'M':0, 'F':1, 'I':2})
data = df_data

# 결측치 20% 생성
missing_length = 0.2
for col in train_col:
    nan_mask = np.random.rand(data.shape[0]) < missing_length
    data.loc[nan_mask, col] = np.nan

data_with_missing = data

# 반복 횟수 설정
num_iterations = 10

accuracy_list = []

for iteration in range(num_iterations):
    # Train set과 test set으로 분할
    train_data, test_data = train_test_split(data_with_missing, test_size=0.2, random_state=iteration)

    # 데이터 결측치 채우기
    df_train, df_test = datawig.utils.random_split(train_data)
    imputer = datawig.SimpleImputer(
        input_columns= train_col,
        output_column='class',
        output_path='imputer_model'
    )
    imputer.fit(train_df=df_train, num_epochs=50)
    train_data = imputer.predict(train_data)
    print(" ==== imputation train_Data ====", train_data)
    test_data = imputer.predict(test_data)
    print(" ==== imputation test_data ====", test_data)
    # 학습을 위한 데이터 준비
    train_X = train_data.drop(columns=['class'])
    train_y = pd.get_dummies(train_data['class'])
    test_X = test_data.drop(columns=['class'])
    test_y = pd.get_dummies(test_data['class'])

    # 신경망 모델 초기화 및 학습
    model = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=train_y.shape[1], train_X=train_X)
    num_epochs = 50
    batch_size = 32

    model.train_model(train_X, train_y, num_epochs, batch_size)
    accuracy = model.get_accuracy(test_X.values, test_y.values)
    print("==========================================")
    print(str(iteration+1)+"th accuracy === : ", accuracy)
    print("==========================================")
    accuracy_list.append(accuracy)
    model.sess.close()

# 평균과 표준편차 계산
accuracy_mean = np.mean(accuracy_list)
accuracy_std = np.std(accuracy_list)

# 결과 출력
print("==========================================")
print("Mean Accuracy: {:.2f}".format(accuracy_mean))
print("Standard Deviation of Accuracy: {:.2f}".format(accuracy_std))
print("======== result : {:.4f} ± {:.4f}".format(sum(accuracy_list)/len(accuracy_list), np.std(accuracy_list)))
print("==========================================")