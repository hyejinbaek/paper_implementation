## Multi-model ensemble method
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import os
from setproctitle import setproctitle
# CUDA 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 프로세스 제목 설정
setproctitle('hyejin')

# 가상의 데이터셋을 사용하고자 가정
# 실제 데이터셋을 사용할 경우 데이터 로드 및 전처리 과정이 필요
# 아래는 예시로 사용될 수 있는 가상의 데이터셋과 모델들
# 실제 코드에서는 데이터 로딩 및 모델 트레이닝을 위한 데이터 전처리 등이 필요

# 가상의 데이터셋 생성
# X_train, X_test, y_train, y_test = ...
data_pth = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/missing/1_breast.csv'
df_data = pd.read_csv(data_pth)

prepro_data = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing/1_breast.csv'
prepro_data = pd.read_csv(prepro_data)
prepro_data.columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']

train_col =['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']

data_with_missing = df_data
print(data_with_missing)

x = data_with_missing[train_col]
y = data_with_missing['Class']


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

accuracy_list = []
rmse_list = []  # RMSE 값을 저장할 리스트 추가
imputers = {}

# 데이터 전처리 - KNN Imputer 사용
imputer = KNNImputer(n_neighbors=2)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 데이터 분할 - 70% 트레이닝, 30% 테스트
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_imputed, y_train, test_size=0.2, random_state=42)

# 각 모델 생성
model_xgb = XGBClassifier()
model_rf = RandomForestClassifier()
model_etc = ExtraTreesClassifier()

# 모델 트레이닝
model_xgb.fit(X_train_final, y_train_final)
model_rf.fit(X_train_final, y_train_final)
model_etc.fit(X_train_final, y_train_final)

# 앙상블 모델 생성
ensemble_model = VotingClassifier(estimators=[('xgb', model_xgb), ('rf', model_rf), ('etc', model_etc)], voting='soft')
print("1 === ensemble_model ===",ensemble_model)
# 앙상블 모델 트레이닝
ensemble_model.fit(X_train_final, y_train_final)
print("ensemble_model ======= ",ensemble_model)
# 앙상블 모델 예측
y_pred = ensemble_model.predict(X_val)
print("y_pred -------- ", y_pred)
# 정확도 측정
accuracy = accuracy_score(y_val, y_pred)

# 결과 출력
print(f"Accuracy: {accuracy}")

# 모델 예측값
y_pred_regression = ensemble_model.predict(X_val)

# RMSE 계산
rmse = np.sqrt(mean_squared_error(y_val, y_pred_regression))
print(f"RMSE: {rmse}")