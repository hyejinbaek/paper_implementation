## Multi-model ensemble method

from sklearn.model_selection import train_test_split
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

from sklearn.impute import KNNImputer
from datawig import SimpleImputer

# CUDA 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 프로세스 제목 설정
setproctitle('hyejin')

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/res/RMSE/22_seeds_ensemble_method_res.csv'

# 결과를 저장할 리스트 초기화
results = []

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
prepro_data = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing/22_seeds.csv'
prepro_data = pd.read_csv(prepro_data)

data_pth = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/missing/22_seeds.csv'
df_data = pd.read_csv(data_pth)
train_col = ['area', 'perimeter', 'compactness', 'length(kernel)', 'width(kernel)', 'asymmetry', 'length(kernel_groove)']
prepro_x = prepro_data[train_col]
prepro_y = prepro_data['class']

data_with_missing = df_data

x = data_with_missing[train_col]
y = data_with_missing['class']

# 반복 횟수 설정
num_iterations = 30

accuracy_list = []
rmse_list = []  # RMSE 값을 저장할 리스트 추가
imputers = {}

for iteration in range(num_iterations):
    # Train set과 test set으로 분할
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

    # DataWig
    x_trnval_datawig = pd.DataFrame(X_train)
    y_trnval_datawig = pd.DataFrame(y_train)
    x_tst_datawig = pd.DataFrame(X_test)
    y_tst_datawig = pd.DataFrame(y_test)

    for col in train_col:
        imputer = SimpleImputer(
            input_columns=train_col,
            output_column=col,
            output_path=f'./imputer_model/imputer_model_{col}'
        )
        imputer.fit(train_df=x_trnval_datawig, num_epochs=5)
        imputers[col] = imputer

    # Impute missing values for each column in train_data
    train_imputed_data = {}

    for col, imputer in imputers.items():
        predictions = imputer.predict(x_trnval_datawig)
        train_imputed_data[col] = predictions[col + '_imputed']

    # Create a DataFrame with imputed values for train set
    train_imputed_df = pd.DataFrame(train_imputed_data)

    # Impute missing values for each column in test_data
    test_imputed_data = {}
    for col, imputer in imputers.items():
        predictions = imputer.predict(x_tst_datawig)
        test_imputed_data[col] = predictions[col + '_imputed']

    # Create a DataFrame with imputed values for test set
    test_imputed_df = pd.DataFrame(test_imputed_data)

    # datawig imputation을 위해 데이터 프레임으로 전환
    x_trnval_datawig_imputed = train_imputed_df[train_col]
    y_trnval_datawig_imputed = y_trnval_datawig
    x_tst_datawig_imputed = test_imputed_df[train_col]
    y_tst_datawig_imputed = y_tst_datawig

    # Knn Imputer
    knn_imputer = KNNImputer(n_neighbors=5)

    # Knn Imputation
    train_data_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(X_train), columns=train_col)
    test_data_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(X_test), columns=train_col)

    # knn imputation 데이터 준비
    train_X_knn_imputed = train_data_knn_imputed
    train_y_knn_imputed = y_train
    test_X_knn_imputed = test_data_knn_imputed
    test_y_knn_imputed = y_test

    # Stacked ensemble method
    imputer = KNNImputer()
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Stacked ensemble method
    # 각각의 분류기를 독립적으로 학습시키고 예측한다고 가정
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train_imputed, y_train)
    xgb_pred_proba = xgb_model.predict_proba(X_test_imputed)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_imputed, y_train)
    rf_pred_proba = rf_model.predict_proba(X_test_imputed)

    etc_model = ExtraTreesClassifier()
    etc_model.fit(X_train_imputed, y_train)
    etc_pred_proba = etc_model.predict_proba(X_test_imputed)

    # 각 모델의 예측 확률을 결합하여 최종 예측을 생성한다
    ensemble_pred_proba = (xgb_pred_proba + rf_pred_proba + etc_pred_proba) / 3
    # print(" === ensemble_pred_proba == ", ensemble_pred_proba)

    # 신경망 모델 초기화 및 학습 (datawig Imputation)
    model_datawig_imputation = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))
    model_datawig_imputation.train_model(x_trnval_datawig_imputed, y_trnval_datawig_imputed, num_epochs=50, batch_size=32)
    accuracy_datawig_imputation = model_datawig_imputation.get_accuracy(x_tst_datawig_imputed.values, y_tst_datawig_imputed.values.reshape(-1, 1))

    # 신경망 모델 초기화 및 학습 (knn Imputation)
    model_knn_imputation = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))
    model_knn_imputation.train_model(train_X_knn_imputed, train_y_knn_imputed, num_epochs=50, batch_size=32)
    accuracy_knn_imputation = model_knn_imputation.get_accuracy(test_X_knn_imputed.values, test_y_knn_imputed.values.reshape(-1, 1))

    # 신경망 모델 학습
    train_imputed_df = pd.DataFrame(X_train_imputed)
    test_imputed_df = pd.DataFrame(X_test_imputed)
    multi_model = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))  # Pass num_features
    multi_model.train_model(train_imputed_df, y_train, num_epochs=50, batch_size=32)
    # accuracy_multi_imputation = multi_model.get_accuracy(X_test_imputed, y_test)

    # 모델 학습 후 imputation 결과 확인
    datawig_imputed_model = model_datawig_imputation.sess.run(model_datawig_imputation.pred, feed_dict={model_datawig_imputation.x: test_imputed_df.values})
    knn_imputed_model = model_knn_imputation.sess.run(model_knn_imputation.pred, feed_dict={model_knn_imputation.x: test_X_knn_imputed.values})
    multi_imputed_model = multi_model.sess.run(multi_model.pred, feed_dict={multi_model.x: X_test_imputed})

    # 예측값 평균 계산
    avg_predictions = (datawig_imputed_model + knn_imputed_model + multi_imputed_model) / 3

    # accuracy 계산
    ensemble_accuracy = accuracy_score(y_test.values, np.round(avg_predictions))
    accuracy_list.append(ensemble_accuracy)
    ensemble_accuracy_std = np.std(accuracy_list)

    # 결측치 생성 전의 데이터를 동일하게 train/test로 나누어서 저장
    original_x_train, original_x_test, original_y_train, original_y_test = train_test_split(prepro_x, prepro_y, test_size=0.2, random_state=iteration)
    
    # RMSE 계산
    rmse = sqrt(((original_y_test.values - avg_predictions.flatten()) ** 2).mean())

    # RMSE의 표준편차 계산
    rmse_list.append(rmse)
    rmse_std = np.std(rmse_list)

    print("==========================================")
    print(str(iteration + 1) + "th Prediction Average : ", avg_predictions)
    print(str(iteration + 1) + "th Ensemble Accuracy : {:.4f} ± {:.4f}".format(ensemble_accuracy, ensemble_accuracy_std))
    print(str(iteration + 1) + "th Ensemble RMSE : {:.4f} ± {:.4f}".format(rmse, rmse_std))
    print("==========================================")


     # 결과를 딕셔너리로 저장
    result = {
        'Dataset' : '22_seeds',
        'method' : '20_multi+knn+datawig',
        'Experiment': iteration + 1,
        'Accuracy': "{:.4f} ± {:.4f}".format(np.mean(accuracy_list), np.std(accuracy_list)),
        'RMSE': "{:.4f} ± {:.4f}".format(rmse, rmse_std)

    }
    results.append(result)

print("==========================================")
print("=== Accuracy result : {:.4f} ± {:.4f}".format(sum(accuracy_list)/len(accuracy_list), np.std(accuracy_list)))
print("=== RMSE result : {:.4f} ± {:.4f}".format(sum(rmse_list)/len(rmse_list), np.std(rmse_list)))
print("==========================================")

# 결과를 DataFrame으로 변환하여 CSV 파일에 추가로 저장
results_df = pd.DataFrame(results)
if os.path.exists(result_csv_path):
    results_df.to_csv(result_csv_path, mode='a', header=False, index=False)
else:
    results_df.to_csv(result_csv_path, index=False)

print("Results saved to:", result_csv_path)