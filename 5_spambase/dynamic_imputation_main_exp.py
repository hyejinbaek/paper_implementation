# 데이터셋 변경하여 진행(spambase dataset)
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
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/res/add_rmse/5_spambase_ensemble_method_res.csv'

# 결과를 저장할 리스트 초기화
results = []
rmse_list = []

def main(args):

    seed = args.seed
    missing_rate = args.missing_rate
    
    hyperparameters = {'num_mi': args.num_mi, 'm': args.m, 'tau': args.tau}

    prepro_data = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing/5_spambase.csv'
    prepro_data = pd.read_csv(prepro_data)
    train_col = ['word_freq_make','word_freq_address','mword_freq_all','word_freq_3d','word_freq_our','word_freq_over','word_freq_remove','word_freq_internet','word_freq_order','word_freq_mail',
                'word_freq_receive','word_freq_will','word_freq_people','word_freq_report','word_freq_addresses','word_freq_free','word_freq_business','word_freq_email','word_freq_you','word_freq_credit',
                'word_freq_your','word_freq_font','word_freq_000','word_freq_money','word_freq_hp','word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet','word_freq_857',
                'word_freq_data','word_freq_415','word_freq_85','word_freq_technology','word_freq_1999','word_freq_parts','word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original',
                'word_freq_project','word_freq_re','word_freq_edu','word_freq_table','word_freq_conference','char_freq_1','char_freq_2','char_freq_3','char_freq_4','char_freq_5','char_freq_6','capital_run_length_average',
                'capital_run_length_longest','capital_run_length_total']
    prepro_x = prepro_data[train_col]
    prepro_y = prepro_data['class']
    data_pth = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/missing/5_spambase.csv'
    df_data = pd.read_csv(data_pth)
    
    # for문에서 뺌
    x, y = preprocessing(df_data[train_col].values,df_data['class'].values, missing_rate, seed)

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
        #auroc = model.get_auroc(x_tst, y_tst)
        acc_list.append(acc)

        imputed_train_data = model.impute_data(x_trnval)
        # print("Imputed Data for Experiment {}: {}".format(i+1, imputed_train_data))
        # print(imputed_train_data)
        imputed_test_data = model.impute_data(x_tst)
        # print("Imputed Data for Experiment {}: {}".format(i+1, imputed_test_data))
        # print(imputed_test_data)

        # 결측치 생성 전의 데이터를 동일하게 train/test로 나누어서 저장
        original_x_train, original_x_test, original_y_train, original_y_test = train_test_split(prepro_x, prepro_y, test_size=0.2, random_state=i)
        
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
            'Dataset' : '5_spambase',
            'method' : 'dynamic',
            'Experiment': i + 1,
            'Accuracy': "{:.4f} ± {:.4f}".format(acc, np.std(acc)),
            'RMSE': "{:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list))
            }
        results.append(result)


    print("==========================================")
    print("=== result : {:.4f} ± {:.4f}".format(sum(acc_list)/len(acc_list), np.std(acc_list)))
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