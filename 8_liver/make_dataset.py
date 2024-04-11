import pandas as pd
import numpy as np
# 데이터 파일 경로 설정
data_pth = './bupa.data'

# 데이터 불러오기
df_data = pd.read_csv(data_pth)
col_data = df_data.columns = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'class', 'selector']
train_col = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'selector']
data = df_data



missing_length = 0.2
for col in train_col:
    nan_mask = np.random.rand(data.shape[0]) < missing_length
    data.loc[nan_mask, col] = np.nan

data_with_missing = data
data_with_missing.to_csv("liver.csv", index = False)
print("done")