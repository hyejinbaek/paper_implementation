## Multi-model ensemble method
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import tensorflow as tf
from setproctitle import setproctitle
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from math import sqrt
# CUDA 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 프로세스 제목 설정
setproctitle('hyejin')

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/res/multi_method/10_lymphography_ensemble_method_res.csv'

# 결과를 저장할 리스트 초기화
results = []


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
        train_X = pd.DataFrame(train_X)  # Convert train_X to Pandas DataFrame
        train_y = pd.DataFrame(train_y)
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

            acc = self.sess.run(accuracy, feed_dict={self.x: x_tst, self.y_true: np.array(y_tst).reshape(-1, 1)})  # 수정된 부분
        else:
            y_tst_hat = self.sess.run(self.pred, feed_dict={self.x: x_tst})
            y_tst_hat = np.argmax(y_tst_hat, axis=1)

            acc = accuracy_score(np.argmax(y_tst, axis=1), y_tst_hat)

        return acc

    
    
# 데이터 파일 경로 설정
data_pth = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/missing/10_lymphography.csv'
df_data = pd.read_csv(data_pth)
train_col =['lymphatics', 'block of affere', 'bl. of lymph.c', 'bl. of lymph.s', 'by pass', 'extravasates', 'regeneration of', 'early uptake in',
                    'lym.nodes dimin', 'lym.nodes enlar', 'changes in lym', 'defect in node', 'changes in node', 'changes in stru', 'special forms', 'dislocation of',
                    'exclusion of no', 'no. of nodes in']

prepro_data = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing/10_lymphography.csv'
prepro_data = pd.read_csv(prepro_data)

prepro_x = prepro_data[train_col]
prepro_y = prepro_data['class']

data_with_missing = df_data
print(" === data_with_missing ===", data_with_missing)

x = data_with_missing[train_col]
y = data_with_missing['class']
print(" === x ===", x)
print(" === y ===", y)

# 반복 횟수 설정
num_iterations = 30

accuracy_list = []
rmse_list = []  # RMSE 값을 저장할 리스트 추가
imputers = {}

for iteration in range(num_iterations):
    # Train set과 test set으로 분할
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    print(" === X_train ===", X_train)
    print(" === X_test ===", X_test)
    print(" === y_train ===", y_train)
    print(" === y_test ===", y_test)

    imputer = KNNImputer()
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # PCA를 사용하여 중요한 특성을 선택한다고 가정
    pca = PCA(n_components=min(X_train_imputed.shape[1], X_train_imputed.shape[0]))  # 특성 수와 동일하게 설정
    X_train_pca = pca.fit_transform(X_train_imputed)
    X_test_pca = pca.transform(X_test_imputed)
    print(" === X_train_pca ===", X_train_pca)
    print(" === X_test_pca ===", X_test_pca)

    # 2. 모델 앙상블
    # 각각의 분류기를 독립적으로 학습시키고 예측한다고 가정
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train_pca, y_train)
    xgb_pred_proba = xgb_model.predict_proba(X_test_pca)
    print(" === xgb_pred_proba ===", xgb_pred_proba)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_pca, y_train)
    rf_pred_proba = rf_model.predict_proba(X_test_pca)
    print(" === rf_pred_proba ===", rf_pred_proba)

    etc_model = ExtraTreesClassifier()
    etc_model.fit(X_train_pca, y_train)
    etc_pred_proba = etc_model.predict_proba(X_test_pca)
    print(" === etc_pred_proba ===", etc_pred_proba)

    # 각 모델의 예측 확률을 결합하여 최종 예측을 생성한다
    ensemble_pred_proba = (xgb_pred_proba + rf_pred_proba + etc_pred_proba) / 3
    print(" === ensemble_pred_proba ===", ensemble_pred_proba)

    # 최종 예측 클래스를 선택한다
    ensemble_pred = np.argmax(ensemble_pred_proba, axis=1)
    print(" === ensemble_pred ===", ensemble_pred)

    # 평가 지표인 accuracy를 계산한다
    accuracy = accuracy_score(y_test, ensemble_pred)
    print("==========================================")
    print(str(iteration+1)+"th accuracy === : ", accuracy)
    print("==========================================")

    # 신경망 모델 학습
    model = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))  # Pass num_features
    num_epochs = 50
    batch_size = 32
    model.train_model(X_train_imputed, y_train, num_epochs, batch_size)

    accuracy_imputed = model.get_accuracy(X_test_imputed, y_test)
    print("Accuracy for imputed data:", accuracy_imputed)


    print("==========================================")
    print(" ==== neural network 추가한 뒤 accuracy ")
    print(str(iteration+1)+"th accuracy === : ", accuracy_imputed)
    print("==========================================")
    accuracy_list.append(accuracy_imputed)

    model.sess.close()

    # 모든 반복이 끝난 후에 평균 및 표준편차 계산
    accuracy_mean = np.mean(accuracy_list)
    accuracy_std = np.std(accuracy_list)

    original_x_train, original_x_test, original_y_train, original_y_test = train_test_split(prepro_x, prepro_y, test_size=0.2, random_state=iteration)
    ensemble_pred_original = ensemble_pred_proba.argmax(axis=1)
    original_y_test.reset_index(drop=True, inplace=True)
    comparison = pd.DataFrame({'Original': original_y_test, 'Predicted': ensemble_pred_original})
    print(comparison)

    original_y_test.reset_index(drop=True, inplace=True)
    rmse = sqrt(mean_squared_error(original_y_test, ensemble_pred_original))
    rmse_list.append(rmse)

    print("==========================================")
    print(" ==== neural network 추가한 뒤 rmse ")
    print(str(iteration+1)+"th rmse === : ", rmse)
    print("==========================================")

     # 결과를 딕셔너리로 저장
    result = {
        'Dataset' : '10_lympho',
        'method' : 'multi',
        'Experiment': iteration + 1,
        'Accuracy': "{:.4f} ± {:.4f}".format(accuracy, np.std(accuracy)),
        'RMSE': "{:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list))

    }
    results.append(result)

print("==========================================")
print("=== result : {:.4f} ± {:.4f}".format(sum(accuracy_list)/len(accuracy_list), np.std(accuracy_list)))
print("=== RMSE result : {:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)))
print("==========================================")

# 결과를 DataFrame으로 변환하여 CSV 파일에 추가로 저장
results_df = pd.DataFrame(results)
if os.path.exists(result_csv_path):
    results_df.to_csv(result_csv_path, mode='a', header=False, index=False)
else:
    results_df.to_csv(result_csv_path, index=False)

print("Results saved to:", result_csv_path)