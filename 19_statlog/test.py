import numpy as np
import pandas as pd


data_pth = './german-numeric.data'
# sep='\s+'는 데이터를 공백 문자를 기준으로 분리하도록 지정
df_data = pd.read_csv(data_pth, sep='\s+', header=None)

df_data.columns = ['at1', 'at2', 'at3', 'at4', 'at5', 'at6', 'at7', 'at8', 'at9', 'at10', 'at11', 'at12', 'at13', 'at14',
                   'at15', 'at16', 'at17', 'at18', 'at19', 'at20', 'at21', 'at22', 'at23', 'at24','class']
train_col = ['at1', 'at2', 'at3', 'at4', 'at5', 'at6', 'at7', 'at8', 'at9', 'at10', 'at11', 'at12', 'at13', 'at14',
                   'at15', 'at16', 'at17', 'at18', 'at19', 'at20', 'at21', 'at22', 'at23', 'at24']

print(df_data.columns)
print(df_data)