# 데이터셋 변경하여 진행(breast-cancer dataset)
# tensorflow version : 2.12.0
# 실행 명령어 : python test.py --seed 0 --missing_rate 20 --num_mi 5 --m 10 --tau 0.05
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from setproctitle import *
setproctitle('hyejin')
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from dynamic_imputation_model import Dynamic_imputation_nn
from dynamic_imputation_preprocessing import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 데이터 파일 경로 설정
data_pth = './abalone.data'

# 데이터 불러오기
df_data = pd.read_csv(data_pth)
col_data = df_data.columns = ['class','Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
train_col =  ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
df_data['class'] = df_data['class'].replace({'M':0, 'F':1, 'I':2})

data = df_data
print(" == data === ", data)
print(data['Rings'].mean())

# 고정 !!
if len(data)>10000:
    np.random.seed(0)
    random_sampled_idx = np.random.choice(len(data), 10000, replace=False)
    data = data[random_sampled_idx]
    

# "Rings" 값이 특정 임계값 이상이면 1, 그렇지 않으면 0으로 설정
threshold = 9  # 예를 들어, 임계값을 10으로 설정
# "class" 열을 기반으로 "y" 값을 생성
data['y'] = (data['class'] >= threshold).astype(int)
print(" == data['y'] === ",data['y'])
# "y" 값을 (n,1) 형태로 변환
y = data['y'].values.reshape(-1, 1)
print(" === y ===", y)

x = data[train_col].values
print(" == x === ", x)