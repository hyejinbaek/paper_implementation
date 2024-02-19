## Multi-model ensemble method
## nn 제외
from sklearn.decomposition import PCA
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

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# CUDA 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 프로세스 제목 설정
setproctitle('hyejin')

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/res/multi_method/5_spambase_ensemble_method_res.csv'

# 결과를 저장할 리스트 초기화
results = []

    
# 데이터 파일 경로 설정
data_pth = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/missing/5_spambase.csv'
df_data = pd.read_csv(data_pth)
train_col = ['word_freq_make','word_freq_address','mword_freq_all','word_freq_3d','word_freq_our','word_freq_over','word_freq_remove','word_freq_internet','word_freq_order','word_freq_mail',
'word_freq_receive','word_freq_will','word_freq_people','word_freq_report','word_freq_addresses','word_freq_free','word_freq_business','word_freq_email','word_freq_you','word_freq_credit',
'word_freq_your','word_freq_font','word_freq_000','word_freq_money','word_freq_hp','word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet','word_freq_857',
'word_freq_data','word_freq_415','word_freq_85','word_freq_technology','word_freq_1999','word_freq_parts','word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original',
'word_freq_project','word_freq_re','word_freq_edu','word_freq_table','word_freq_conference','char_freq_1','char_freq_2','char_freq_3','char_freq_4','char_freq_5','char_freq_6','capital_run_length_average',
'capital_run_length_longest','capital_run_length_total']

prepro_data = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing/5_spambase.csv'
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
    print(str(iteration+1)+"th rmse === : ", rmse)
    print("==========================================")

     # 결과를 딕셔너리로 저장
    result = {
        'Dataset' : '5_spambase',
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