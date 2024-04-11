import numpy as np
import pandas as pd


data_pth = './cmc.data'

df_data = pd.read_csv(data_pth, sep=',', header=None)

df_data.columns = ['age', 'wife education', 'husband education', 'number', 'wife religion', 'wife working', 'husband occupation', 'standard', 'media','class']
train_col = ['age', 'wife education', 'husband education', 'number', 'wife religion', 'wife working', 'husband occupation', 'standard', 'media']

print(df_data)