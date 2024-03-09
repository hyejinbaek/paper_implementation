## Multi-model ensemble method
## nn 제외
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

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from net import ShuffleNetV2

# CUDA 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 프로세스 제목 설정
setproctitle('hyejin')

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/res/multi_method/2_heart_ensemble_method_res.csv'

# 결과를 저장할 리스트 초기화
results = []
    
# 데이터 파일 경로 설정
data_pth = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/missing/2_heart.csv'
df_data = pd.read_csv(data_pth)
train_col = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

prepro_data = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing/2_heart.csv'
prepro_data = pd.read_csv(prepro_data)
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
accuracy_net_list = []

for iteration in range(num_iterations):
    # Train set과 test set으로 분할
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy='most_frequent')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.fit_transform(X_test)

    # Data standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.fit_transform(X_test_imputed)

    # PCA 진행
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.fit_transform(X_test_scaled)

    # PCA 역변환을 수행하여 원래 feature 공간으로 되돌린다.
    X_train_imputed_inv = pca.inverse_transform(X_train_pca)
    X_test_imputed_inv = pca.inverse_transform(X_test_pca)
    
    # 표준화된 데이터를 다시 되돌린다.
    X_train_filled = scaler.inverse_transform(X_train_imputed_inv)
    X_test_filled = scaler.inverse_transform(X_test_imputed_inv)

    # 2. 모델 앙상블
    # 각각의 분류기를 독립적으로 학습시키고 예측한다고 가정
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train_filled, y_train)
    xgb_pred_proba = xgb_model.predict_proba(X_test_filled)


    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_filled, y_train)
    rf_pred_proba = rf_model.predict_proba(X_test_filled)

    etc_model = ExtraTreesClassifier()
    etc_model.fit(X_train_filled, y_train)
    etc_pred_proba = etc_model.predict_proba(X_test_filled)


    # 각 모델의 예측 확률을 결합하여 최종 예측을 생성한다
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
    print(str(iteration+1)+"th accuracy === : ", rmse)
    print("==========================================")

    # shuffleNet 추가
    # Model hyperparameters
    num_classes = 2
    model_scale = 2.0
    shuffle_group = 2

    # Model: Using Net1 from net.py 
    net = ShuffleNetV2(cls=num_classes, model_scale=model_scale, shuffle_group=shuffle_group)
    print(net)
    # Define loss and optimization operations for the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    with tf.GradientTape() as tape:
        logits_net = net(X_train_filled, training=True)
        loss_net = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=logits_net))
    gradients_net = tape.gradient(loss_net, net.trainable_variables)
    optimizer.apply_gradients(zip(gradients_net, net.trainable_variables))
    # print(f'Epoch {iteration + 1}, Loss Net: {loss_net.numpy()}')
    print(f'======== Epoch {iteration + 1}, Loss Net: {loss_net}')

    # Evaluation on the test set
    # accuracy_net = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net(X_test_filled, training=False), axis=1), y_test), tf.float32)).numpy()
    accuracy_net = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net(X_test_filled, training=False), axis=1), y_test), tf.float32))
    accuracy_net_value = tf.keras.backend.eval(accuracy_net)
    accuracy_net_std_value = tf.keras.backend.eval(tf.math.reduce_std(accuracy_net))
    
    print("==========================================")
    print(f'=========== Accuracy Net on Test Set: {accuracy_net_value}')
    print("=== Shuffle_Accuracy : {:.4f} ± {:.4f}".format(accuracy_net_value, accuracy_net_std_value))
    print("==========================================")
    
    accuracy_net_list.append(accuracy_net_value)


     # 결과를 딕셔너리로 저장
    result = {
        'Dataset' : '2_heart',
        'method' : 'multi(nn제외)',
        'Experiment': iteration + 1,
        'Accuracy': "{:.4f} ± {:.4f}".format(accuracy, np.std(accuracy)),
        'RMSE': "{:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)),
        'Shuffle_Accuracy' : "{:.4f} ± {:.4f}".format(accuracy_net_value, accuracy_net_std_value)

    }
    results.append(result)

print("==========================================")
print("=== result : {:.4f} ± {:.4f}".format(sum(accuracy_list)/len(accuracy_list), np.std(accuracy_list)))
print("=== RMSE result : {:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)))
print("==========================================")

# # 결과를 DataFrame으로 변환하여 CSV 파일에 추가로 저장
# results_df = pd.DataFrame(results)
# if os.path.exists(result_csv_path):
#     results_df.to_csv(result_csv_path, mode='a', header=False, index=False)
# else:
#     results_df.to_csv(result_csv_path, index=False)

# print("Results saved to:", result_csv_path)