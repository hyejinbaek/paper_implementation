## multi+knn
from sklearn.impute import KNNImputer
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

from net_test import ShuffleNetV2

# CUDA 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 프로세스 제목 설정
setproctitle('hyejin')

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/res/shuffle/1_breast_ensemble_method_res.csv'

# 결과를 저장할 리스트 초기화
results = []
    
    
# 데이터 파일 경로 설정
data_pth = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/missing/1_breast.csv'
df_data = pd.read_csv(data_pth)
train_col =['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']

prepro_data = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing/1_breast.csv'
prepro_data = pd.read_csv(prepro_data)
prepro_data.columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
prepro_x = prepro_data[train_col]
prepro_y = prepro_data['Class']

data_with_missing = df_data

x = data_with_missing[train_col]
y = data_with_missing['Class']

# 반복 횟수 설정
num_iterations = 30

accuracy_list = []
rmse_list = []  # RMSE 값을 저장할 리스트 추가
imputers = {}

accuracy_net_list = []

# shuffleNet 추가
# Model hyperparameters
num_classes = 2
model_scale = 1.0
shuffle_group = 2

# Model: Using Net1 from net.py 
net = ShuffleNetV2(cls=num_classes, model_scale=model_scale, shuffle_group=shuffle_group)
print(net)
# Define loss and optimization operations for the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

for iteration in range(num_iterations):
    
    
    # Train set과 test set으로 분할
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

    ### 1. Knn
    # 데이터 결측치 채우기 (KNN Imputation)
    x_trnval_knn = pd.DataFrame(X_train, columns=train_col)
    y_trnval_knn = pd.DataFrame(y_train, columns=['Class'])
    x_tst_knn = pd.DataFrame(X_test, columns=train_col)
    y_tst_knn = pd.DataFrame(y_test, columns=['Class'])
    
    knnimputer = KNNImputer(n_neighbors=5)
    train_knn_data = pd.DataFrame(knnimputer.fit_transform(x_trnval_knn), columns=X_train.columns)
    test_knn_data = pd.DataFrame(knnimputer.transform(x_tst_knn), columns=X_test.columns)

    # knn imputation 학습 위한 데이터 준비
    train_X_knn_imputed = train_knn_data
    train_y_knn_imputed = y_trnval_knn
    test_X_knn_imputed = test_knn_data
    test_y_knn_imputed = y_tst_knn

    ### 2. Multi-ensemble
    imputer = KNNImputer()
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # 2. 모델 앙상블
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
    ## class 예측으로 잘못된 방향으로 구현됨. feature imputation으로 변경필요
    ensemble_pred_proba = (xgb_pred_proba + rf_pred_proba + etc_pred_proba) / 3

    # 최종 예측 클래스를 선택한다
    ensemble_pred = np.argmax(ensemble_pred_proba, axis=1)


    # 평가 지표인 accuracy를 계산한다
    accuracy = accuracy_score(y_test, ensemble_pred)
    print("==========================================")
    print(str(iteration+1)+"th accuracy === : ", accuracy)
    print("==========================================")

    
    accuracy_list.append(accuracy)


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
    print(str(iteration+1)+"th rmse === : ", rmse)
    print("==========================================")
    
    for epoch in range(50):
        with tf.GradientTape() as tape:
            logits_net = net(X_train_imputed, training=True)
            loss_net = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=logits_net))
        gradients_net = tape.gradient(loss_net, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients_net, net.trainable_variables))
        loss_net_value = tf.keras.backend.eval(loss_net)
        print(f'Epoch {epoch + 1}, Loss Net: {loss_net_value}')



    # Evaluation on the test set
    # accuracy_net = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net(X_test_filled, training=False), axis=1), y_test), tf.float32)).numpy()
    accuracy_net = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net(X_test_imputed, training=False), axis=1), y_test), tf.float32))
    accuracy_net_value = tf.keras.backend.eval(accuracy_net)
    accuracy_net_std_value = tf.keras.backend.eval(tf.math.reduce_std(accuracy_net))
    
    print("==========================================")
    print(f'=========== Accuracy Net on Test Set: {accuracy_net_value}')
    print("=== Shuffle_Accuracy : {:.4f} ± {:.4f}".format(accuracy_net_value, accuracy_net_std_value))
    print("==========================================")
    
    accuracy_net_list.append((accuracy_net_value, accuracy_net_std_value))

     # 결과를 딕셔너리로 저장
    result = {
        'Dataset' : '1_breast',
        'method' : 'multi-shuffle',
        'Experiment': iteration + 1,
        'Imputation_Accuracy': "{:.4f} ± {:.4f}".format(accuracy, np.std(accuracy)),
        'RMSE': "{:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)),
        'Shuffle_Accuracy' : "{:.4f} ± {:.4f}".format(accuracy_net_value, accuracy_net_std_value)

    }
    results.append(result)

accuracy_net_mean, accuracy_net_std = zip(*accuracy_net_list)

print("==========================================")
print("=== result : {:.4f} ± {:.4f}".format(sum(accuracy_list)/len(accuracy_list), np.std(accuracy_list)))
print("=== RMSE result : {:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)))
print("=== Shuffle result : {:.4f} ± {:.4f}".format(np.mean(accuracy_net_mean), np.mean(accuracy_net_std)))
print("==========================================")

# # 결과를 DataFrame으로 변환하여 CSV 파일에 추가로 저장
# results_df = pd.DataFrame(results)
# if os.path.exists(result_csv_path):
#     results_df.to_csv(result_csv_path, mode='a', header=False, index=False)
# else:
#     results_df.to_csv(result_csv_path, index=False)

# print("Results saved to:", result_csv_path)