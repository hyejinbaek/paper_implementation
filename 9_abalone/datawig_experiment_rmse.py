import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from setproctitle import setproctitle
from datawig import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# CUDA 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 프로세스 제목 설정
setproctitle('hyejin')

# 함수 정의: RMSE 계산
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def label_encode(df, columns):
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for col in columns:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col].astype(str))
    return df_encoded


class DynamicImputationModel:
    def __init__(self, num_layers, num_hidden, dim_y, num_features):  # Pass num_features
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.dim_y = dim_y
        self.num_features = num_features  # Store num_features in the instance
        tf.compat.v1.disable_eager_execution()
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, self.num_features])  # Use self.num_features here

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
            np.random.seed(epoch)
            np.random.shuffle(indices)
            train_X_shuffled = train_X[indices]
            #print("==== train_X_shuffled ====", train_X_shuffled)
            train_y_shuffled = train_y[indices]
            #print("==== train_y_shuffled ====", train_y_shuffled)

            for i in range(num_batches):
                batch_X = train_X_shuffled[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y_shuffled[i * batch_size: (i + 1) * batch_size]

                self.sess.run(self.train_op, feed_dict={self.x: batch_X, self.y_true: batch_y.reshape(-1, 1)})

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


# 결측치 20% 생성
missing_length = 0.2
for col in train_col:
    nan_mask = np.random.rand(data.shape[0]) < missing_length
    data.loc[nan_mask, col] = np.nan

data_with_missing = data

# 반복 횟수 설정
num_iterations = 10

accuracy_list = []
rmse_list = []  # RMSE 값을 저장할 리스트 추가
imputers = {}

for iteration in range(num_iterations):
    # Train set과 test set으로 분할
    train_data, test_data = train_test_split(data_with_missing, test_size=0.2, random_state=iteration)

    # 데이터 결측치 채우기
    for col in train_col:
        imputer = SimpleImputer(
            input_columns=train_col,
            output_column=col
        )
        imputer.fit(train_df=train_data, num_epochs=5)
        imputers[col] = imputer


    # Impute missing values for each column in train_data
    train_imputed_data = {}
    for col, imputer in imputers.items():
        predictions = imputer.predict(train_data)
        train_imputed_data[col] = predictions[col + '_imputed']  # '_imputed' is added by datawig

    # Create a DataFrame with imputed values for train set
    train_imputed_df = pd.DataFrame(train_imputed_data)

    # Impute missing values for each column in test_data
    test_imputed_data = {}
    for col, imputer in imputers.items():
        predictions = imputer.predict(test_data)
        test_imputed_data[col] = predictions[col + '_imputed']  # '_imputed' is added by datawig

    # Create a DataFrame with imputed values for test set
    test_imputed_df = pd.DataFrame(test_imputed_data)

    print("Train Imputed DataFrame =========\n", train_imputed_df)
    print("Test Imputed DataFrame =========\n", test_imputed_df)

    # 학습을 위한 데이터 준비
    train_X = train_imputed_df[train_col].values  # Select only the columns for training
    train_y = train_data['class'].values  # Convert to NumPy array
    test_X = test_imputed_df[train_col].values  # Select only the columns for testing
    test_y = test_data['class'].values  # Convert to NumPy array

    # 신경망 모델 학습
    model = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))  # Pass num_features
    num_epochs = 50
    batch_size = 32
    model.train_model(train_X, train_y, num_epochs, batch_size)

    accuracy = model.get_accuracy(test_X, test_y.reshape(-1, 1))
    print("==========================================")
    print(str(iteration+1)+"th accuracy === : ", accuracy)
    print("==========================================")
    accuracy_list.append(accuracy)

    # RMSE 계산 및 저장
    y_pred = model.sess.run(model.pred, feed_dict={model.x: test_X})

    rmse = calculate_rmse(test_y, y_pred)
    rmse_list.append(rmse)
    print("==========================================")
    print(str(iteration+1)+"th RMSE === : ", rmse)
    print("==========================================")

    model.sess.close()

# 모든 반복이 끝난 후에 평균 및 표준편차 계산
accuracy_mean = np.mean(accuracy_list)
accuracy_std = np.std(accuracy_list)
rmse_mean = np.mean(rmse_list)
rmse_std = np.std(rmse_list)

print("Mean Accuracy: {:.2f}".format(accuracy_mean))
print("Standard Deviation of Accuracy: {:.2f}".format(accuracy_std))
print("==========================================")
print("=== result : {:.4f} ± {:.4f}".format(sum(accuracy_list)/len(accuracy_list), np.std(accuracy_list)))
print("=== RMSE result : {:.4f} ± {:.4f}".format(rmse_mean,rmse_std))
print("==========================================")


