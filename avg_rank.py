import os
import pandas as pd
import numpy as np

df=pd.read_csv("test_acc_rank.csv")

# 행렬 위치 변환
df=df.transpose()

# ±~뒤에꺼까지 싹 다 지우기
df.replace(to_replace=r' ±.*$', value='', regex=True, inplace=True)

# csv 파일로 저장
df.to_csv("transpose_test_exp_result.csv", header=False, encoding = 'cp949')