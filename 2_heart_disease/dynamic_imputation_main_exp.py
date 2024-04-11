# 데이터셋 변경하여 진행(heart-disease)
# tensorflow version : 2.12.0
# 실행 명령어 : python dynamic_imputation_main_exp.py --seed 0 --missing_rate 20 --num_mi 5 --m 10 --tau 0.05
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
from sklearn.preprocessing import MinMaxScaler

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/res/add_rmse/2_heart_ensemble_method_res.csv'

# 결과를 저장할 리스트 초기화
results = []


def main(args):
    rmse_list = []
    seed = args.seed
    #dataset = args.dataset
    missing_rate = args.missing_rate
    
    hyperparameters = {'num_mi': args.num_mi, 'm': args.m, 'tau': args.tau}

    prepro_data = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing/2_heart.csv'
    prepro_data = pd.read_csv(prepro_data)
    train_col = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    prepro_x = prepro_data[train_col]
    prepro_y = prepro_data['class']

    data_pth = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/missing/2_heart.csv'
    df_data = pd.read_csv(data_pth)
    train_col = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

    data = df_data.values
    
    x = data[:,:-1]
    y = data[:,-1]

    # for문에서 뺌
    x,y = preprocessing(x, y, missing_rate, seed)

    acc_list, auroc = [], []

    for i  in range(30):
        x_trnval, x_tst, y_trnval, y_tst = train_test_split(x,y, test_size=0.2, shuffle=True, random_state=i)

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
        acc_list.append(acc)

        imputed_train_data = model.impute_data(x_trnval)
        # print("Imputed Data for Experiment {}: {}".format(i+1, imputed_train_data))
        # print(imputed_train_data)
        imputed_test_data = model.impute_data(x_tst)
        # print("Imputed Data for Experiment {}: {}".format(i+1, imputed_test_data))
        # print(imputed_test_data)


        # 결측치 생성 전의 데이터를 동일하게 train/test로 나누어서 저장
        # original_data_train, original_data_test = train_test_split(prepro_data, test_size=0.2, random_state=i)
        original_x_train, original_x_test, original_y_train, original_y_test = train_test_split(prepro_x, prepro_y, test_size=0.2, random_state=i)
        print(" == original_x_test", original_x_test)
        print(" == imputed_test_data == ", imputed_test_data)

        # Min-Max Scaling 수행
        scaler = MinMaxScaler(feature_range=(-1, 1))  # imputed_test_data와 동일한 범위로 조정
        original_x_test_scaled = scaler.fit_transform(original_x_test)
        print(" == original_x_test_scaled == ", original_x_test_scaled)

        # RMSE 계산 및 리스트에 추가
        rmse = sqrt(mean_squared_error(original_x_test_scaled, imputed_test_data))
        print("==========================================")
        print(str(i + 1) + "th dynamic Imputation rmse: ", rmse)
        print("==========================================")
        rmse_list.append(rmse)
        

        # 결과를 딕셔너리로 저장
        result = {
            'Dataset' : '2_heart',
            'method' : 'dynamic',
            'Experiment': i + 1,
            'Accuracy': "{:.4f} ± {:.4f}".format(acc, np.std(acc)),
            'RMSE': "{:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list))
        }
        results.append(result)



    print("==========================================")
    print("=== Accuracy result : {:.4f} ± {:.4f}".format(sum(acc_list)/len(acc_list), np.std(acc_list)))
    print("=== RMSE result : {:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)))
    print("==========================================")
    
   # 결과를 DataFrame으로 변환하여 CSV 파일에 추가로 저장
    results_df = pd.DataFrame(results)
    if os.path.exists(result_csv_path):
        results_df.to_csv(result_csv_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(result_csv_path, index=False)

    print("Results saved to:", result_csv_path)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Dynamic imputation')
    arg_parser.add_argument('--seed', help='Random seed', default=27407, type= int)
    arg_parser.add_argument('--missing_rate', help='Missing rate of dataset', default=20, type=float)
    arg_parser.add_argument('--num_mi', help='Number of multiple imputation for validation set', default=5, type=int)
    arg_parser.add_argument('--m', help='Number of imputations to calculate imputation uncertainty', default=10, type=int)
    arg_parser.add_argument('--tau', help='Threshold of imputation uncertainty', default=0.05, type=float)
    
    args = arg_parser.parse_args()
    
    main(args)