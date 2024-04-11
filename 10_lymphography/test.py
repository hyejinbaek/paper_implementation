import pandas as pd
import numpy as np

# 데이터 파일 경로 설정
data_pth = './lymphography.data'

# 데이터 불러오기
df_data = pd.read_csv(data_pth)
col_data = df_data.columns = ['class','lymphatics', 'block of affere', 'bl. of lymph.c', 'bl. of lymph.s', 'by pass', 'extravasates', 'regeneration of', 'early uptake in',
                    'lym.nodes dimin', 'lym.nodes enlar', 'changes in lym', 'defect in node', 'changes in node', 'changes in stru', 'special forms', 'dislocation of',
                    'exclusion of no', 'no. of nodes in']
train_col =['lymphatics', 'block of affere', 'bl. of lymph.c', 'bl. of lymph.s', 'by pass', 'extravasates', 'regeneration of', 'early uptake in',
                    'lym.nodes dimin', 'lym.nodes enlar', 'changes in lym', 'defect in node', 'changes in node', 'changes in stru', 'special forms', 'dislocation of',
                    'exclusion of no', 'no. of nodes in']
data = df_data


# 결측치 20% 생성
missing_length = 0.2
for col in train_col:
    nan_mask = np.random.rand(data.shape[0]) < missing_length
    data.loc[nan_mask, col] = np.nan

data_with_missing = data