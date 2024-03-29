import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from setproctitle import setproctitle
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/res/5_spambase_ensemble_method_res.csv'

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
data_pth = './spambase.data'

# 데이터 불러오기
df_data = pd.read_csv(data_pth)
col_data = df_data.columns = ['word_freq_make','word_freq_address','mword_freq_all','word_freq_3d','word_freq_our','word_freq_over','word_freq_remove','word_freq_internet','word_freq_order',
                    'word_freq_mail','word_freq_receive','word_freq_will','word_freq_people','word_freq_report','word_freq_addresses','word_freq_free','word_freq_business',
                    'word_freq_email','word_freq_you','word_freq_credit','word_freq_your','word_freq_font','word_freq_000','word_freq_money','word_freq_hp','word_freq_hpl',
                    'word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet','word_freq_857','word_freq_data','word_freq_415','word_freq_85',
                    'word_freq_technology','word_freq_1999','word_freq_parts','word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original',
                    'word_freq_project','word_freq_re','word_freq_edu','word_freq_table','word_freq_conference','char_freq_;','char_freq_(','char_freq_[','char_freq_!',
                    'char_freq_$','char_freq_#','capital_run_length_average','capital_run_length_longest','capital_run_length_total', 'class']
train_col = ['word_freq_make','word_freq_address','mword_freq_all','word_freq_3d','word_freq_our','word_freq_over','word_freq_remove','word_freq_internet','word_freq_order',
                    'word_freq_mail','word_freq_receive','word_freq_will','word_freq_people','word_freq_report','word_freq_addresses','word_freq_free','word_freq_business',
                    'word_freq_email','word_freq_you','word_freq_credit','word_freq_your','word_freq_font','word_freq_000','word_freq_money','word_freq_hp','word_freq_hpl',
                    'word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet','word_freq_857','word_freq_data','word_freq_415','word_freq_85',
                    'word_freq_technology','word_freq_1999','word_freq_parts','word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original',
                    'word_freq_project','word_freq_re','word_freq_edu','word_freq_table','word_freq_conference','char_freq_;','char_freq_(','char_freq_[','char_freq_!',
                    'char_freq_$','char_freq_#','capital_run_length_average','capital_run_length_longest','capital_run_length_total']

data = df_data


missing_length = 0.2
for col in train_col:
    nan_mask = np.random.rand(data.shape[0]) < missing_length
    data.loc[nan_mask, col] = np.nan

data_with_missing = data

# 반복 횟수 설정
num_iterations = 30

accuracy_list = []

for iteration in range(num_iterations):
    # Train set과 test set으로 분할
    train_data, test_data = train_test_split(data_with_missing, test_size=0.2, random_state=iteration)

    # 데이터 결측치 채우기 (KNN Imputation)
    imputer = KNNImputer(n_neighbors=5)
    train_data_knn_imputed = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
    test_data_knn_imputed = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

    # 데이터 결측치 채우기 (Zero Imputation)
    train_data_zero_imputed = train_data.fillna(0)
    test_data_zero_imputed = test_data.fillna(0)

    # 학습을 위한 데이터 준비
    train_X_knn_imputed = train_data_knn_imputed.drop(columns=['class'])
    train_y_knn_imputed = train_data_knn_imputed['class']
    test_X_knn_imputed = test_data_knn_imputed.drop(columns=['class'])
    test_y_knn_imputed = test_data_knn_imputed['class']

    train_X_zero_imputed = train_data_zero_imputed.drop(columns=['class'])
    train_y_zero_imputed = train_data_zero_imputed['class']
    test_X_zero_imputed = test_data_zero_imputed.drop(columns=['class'])
    test_y_zero_imputed = test_data_zero_imputed['class']

    # 신경망 모델 초기화 및 학습 (KNN Imputation)
    model_knn_imputation = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))
    model_knn_imputation.train_model(train_X_knn_imputed, train_y_knn_imputed, num_epochs=50, batch_size=32)
    accuracy_knn_imputation = model_knn_imputation.get_accuracy(test_X_knn_imputed.values, test_y_knn_imputed.values.reshape(-1, 1))

    # 신경망 모델 초기화 및 학습 (Zero Imputation)
    model_zero_imputation = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))
    model_zero_imputation.train_model(train_X_zero_imputed, train_y_zero_imputed, num_epochs=50, batch_size=32)
    accuracy_zero_imputation = model_zero_imputation.get_accuracy(test_X_zero_imputed.values, test_y_zero_imputed.values.reshape(-1, 1))

    print("==========================================")
    print(str(iteration + 1) + "th KNN Imputation accuracy: ", accuracy_knn_imputation)
    print(str(iteration + 1) + "th Zero Imputation accuracy: ", accuracy_zero_imputation)
    print("==========================================")

    accuracy_list.append(accuracy_knn_imputation)
    accuracy_list.append(accuracy_zero_imputation)

    # 결과를 딕셔너리로 저장 (Ensemble 결과)
    result = {
        'Dataset': '5_spambase',
        'method': '1_zero+knn',
        'Experiment': iteration + 1,
        'Accuracy': "{:.4f} ± {:.4f}".format(np.mean(accuracy_list), np.std(accuracy_list))
    }
    results.append(result)

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
