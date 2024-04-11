import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Embedding, Flatten
from sklearn.preprocessing import LabelEncoder


data_pth = './magic04.data'
df_data = pd.read_csv(data_pth)
# 데이터 불러오기
df_data = pd.read_csv(data_pth)
df_data.columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
train_col = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']
df_data['class'] = df_data['class'].replace({'g':0, 'h':1})
print(df_data.columns)
print(df_data)