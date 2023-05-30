# 데이터셋 변경하여 진행(breast-cancer-wisconsin.data)
# tensorflow version : 2.12.0
# 실행 명령어 : python dynamic_imputation_main.py --seed 0 --missing_rate 30 --num_mi 5 --m 10 --tau 0.05
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from dynamic_imputation_model import Dynamic_imputation_nn
from dynamic_imputation_preprocessing import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import math


def main(args):

    seed = args.seed
    #dataset = args.dataset
    missing_rate = args.missing_rate
    
    hyperparameters = {'num_mi': args.num_mi, 'm': args.m, 'tau': args.tau}

    #data = pd.read_csv('./datasets/{}.csv'.format(dataset), delimiter=',', header=None).values
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    df_data = pd.read_csv(data_url)
    col_data = df_data.columns = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ', 'Single Epithelial Cell Size',
                                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    train_col = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion ',
                'Single Epithelial Cell Size','Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    df_data['Bare Nuclei'] = df_data['Bare Nuclei'].replace('?',0).astype(int)
    df_data['Class'] = df_data['Class'].replace({2:0, 4:1})
    data = df_data[train_col].values
    
    if len(data)>10000:
        np.random.seed(seed)
        random_sampled_idx = np.random.choice(len(data), 10000, replace=False)
        data = data[random_sampled_idx]
    
    x = data[:,:-1]
    y = data[:,-1]
    
    # 교차 검증 코드 추가
    #n_splits = args.n_splits
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    print("==== kf ======", kf)

    acc_list, auroc = [], []
   
    for i  in range(10):
        x_trnval_o, x_tst_o, y_trnval_o, y_tst_o = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = i)

        x_trnval, x_tst, y_trnval, y_tst = preprocessing(x_trnval_o, x_tst_o, y_trnval_o, y_tst_o, missing_rate, seed)

        dim_x = x_trnval.shape[1]

        if y_trnval.shape[1] > 2:
            dim_y = y_trnval.shape[1]
        else:
            dim_y = 1
        save_path = ('./{0}_{1}_model'.format(seed, missing_rate))
        model = Dynamic_imputation_nn(dim_x, dim_y, seed)
        model.train_with_dynamic_imputation(x_trnval, y_trnval, save_path, **hyperparameters)
        acc = model.get_accuracy(x_tst, y_tst)
        print("==========================================")
        print(str(i+1)+"th accuracy === : ", acc)
        print("==========================================")
        #auroc = model.get_auroc(x_tst, y_tst)
        acc_list.append(acc)
    print("==========================================")
    print("mean acc : {}".format(sum(acc_list)/len(acc_list)))
    print("std acc : {}".format(np.std(acc_list)))
    print("==========================================")
    


if __name__ == '__main__':

    # python main.py --seed 0 --dataset avila --missing_rate 30 --num_mi 5 --m 10 --tau 0.05
    # python dynamic_imputation_main.py --seed 0 --missing_rate 30 --num_mi 5 --m 10 --tau 0.05
    arg_parser = argparse.ArgumentParser(description='Dynamic imputation')
    
    arg_parser.add_argument('--seed', help='Random seed', default=27407, type= int)
    #arg_parser.add_argument('--dataset', help='Dataset name', choices=['avila', 'letter'], default=256, type=str)
    arg_parser.add_argument('--missing_rate', help='Missing rate of dataset', default=30, type=float)
    arg_parser.add_argument('--num_mi', help='Number of multiple imputation for validation set', default=5, type=int)
    arg_parser.add_argument('--m', help='Number of imputations to calculate imputation uncertainty', default=10, type=int)
    arg_parser.add_argument('--tau', help='Threshold of imputation uncertainty', default=0.05, type=float)
    
    args = arg_parser.parse_args()
    
    main(args)