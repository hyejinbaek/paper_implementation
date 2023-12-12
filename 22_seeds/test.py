import pandas as pd
import numpy as np


data_pth = './seeds_dataset.txt'

df_data = pd.read_csv(data_pth, sep='\s+', header=None)

df_data.columns = ['area', 'perimeter', 'compactness', 'length(kernel)', 'width(kernel)', 'asymmetry', 'length(kernel_groove)','class']
train_col = ['area', 'perimeter', 'compactness', 'length(kernel)', 'width(kernel)', 'asymmetry', 'length(kernel_groove)']

data = df_data
print(data)