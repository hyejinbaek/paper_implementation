# 데이터셋 변경하여 진행(post-patient dataset)
# tensorflow version : 2.12.0
# 실행 명령어 : python dynamic_imputation_main_exp.py --seed 0 --missing_rate 20 --num_mi 5 --m 10 --tau 0.05
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, Flatten
from math import sqrt


# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/res/14_national_ensemble_method_res.csv'

# 결과를 저장할 리스트 초기화
results = []

def label_encode(df, columns):
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for col in columns:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col].astype(str))
    return df_encoded

def build_embedding_model(input_dims, embedding_dims):
    inputs = []
    embeddings = []
    for input_dim in input_dims:
        input_layer = Input(shape=(1,))
        embedding = Embedding(input_dim, embedding_dims)(input_layer)
        embedding = Flatten()(embedding)
        inputs.append(input_layer)
        embeddings.append(embedding)
    return inputs, embeddings

def main(args):

    seed = args.seed
    missing_rate = args.missing_rate
    
    hyperparameters = {'num_mi': args.num_mi, 'm': args.m, 'tau': args.tau}

    # 데이터 파일 경로 설정
    data_pth = './NHANES_age_prediction.csv'

    # 데이터 불러오기
    df_data = pd.read_csv(data_pth, usecols=lambda column: column != 'SEQN')
    col_data = df_data.columns
    train_col = list(col_data)
    train_col.remove('age_group')
    data = df_data

    # 범주형 피처 선택
    categorical_columns = ['age_group']

    # 레이블 인코딩 적용
    df_encoded = label_encode(df_data, categorical_columns)
    data = df_encoded

    # for문에서 뺌
    x, y = preprocessing(df_encoded[train_col].values,df_encoded['age_group'].values, missing_rate, seed)

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
        print("==========================================")
        print("=== result : {:.4f} ± {:.4f}".format(sum(acc_list)/len(acc_list), np.std(acc_list)))
        print("==========================================")


        # 결과를 딕셔너리로 저장
        result = {
            'Dataset' : '14_national',
            'method' : 'dynamic',
            'Experiment': i + 1,
            'Accuracy': "{:.4f} ± {:.4f}".format(acc, np.std(acc))
        }
        results.append(result)
        
    print("==========================================")
    print("=== result : {:.4f} ± {:.4f}".format(sum(acc_list)/len(acc_list), np.std(acc_list)))
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