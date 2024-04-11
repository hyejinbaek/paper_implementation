import pandas as pd
# ttest_ind : 독립 2- 표본 ttest
# ttest_rel : 1- 표본 ttest 
from scipy.stats import ttest_ind
import numpy as np

# datawig
# knn
# chain01_datawig
# multi

# file_path = './res/chain1/1_breast_datawig_method_res.csv'  # 파일 경로를 실제 파일 경로로 대체
file_path = './res/0_final/21_yeast_ensemble_method_res_final.csv'
data  = pd.read_csv(file_path)

datawig_rmse = data[data['method']== 'datawig']['RMSE'].apply(lambda x: float(x.split(' ± ')[0]))
datawig_rmse_std = data[data['method'] == 'datawig']['RMSE'].apply(lambda x: float(x.split(' ± ')[1]))

knn_rmse = data[data['method']== 'knn']['RMSE'].apply(lambda x: float(x.split(' ± ')[0]))
knn_rmse_std = data[data['method'] == 'knn']['RMSE'].apply(lambda x: float(x.split(' ± ')[1]))

multi_rmse = data[data['method']== 'multi']['RMSE'].apply(lambda x: float(x.split(' ± ')[0]))
multi_rmse_std = data[data['method'] == 'multi']['RMSE'].apply(lambda x: float(x.split(' ± ')[1]))

chain01_datawig_rmse = data[data['method']== 'chain01_datawig']['RMSE'].apply(lambda x: float(x.split(' ± ')[0]))
chain01_datawig_rmse_std = data[data['method'] == 'chain01_datawig']['RMSE'].apply(lambda x: float(x.split(' ± ')[1]))


print("=== DataWig result : {:.4f} ± {:.4f}".format(np.mean(datawig_rmse), np.std(datawig_rmse_std)))
print("=== knn result : {:.4f} ± {:.4f}".format(np.mean(knn_rmse), np.std(knn_rmse_std)))
print("=== multi result : {:.4f} ± {:.4f}".format(np.mean(multi_rmse), np.std(multi_rmse_std)))
print("=== chain01_datawig result : {:.4f} ± {:.4f}".format(np.mean(chain01_datawig_rmse), np.std(chain01_datawig_rmse_std)))
