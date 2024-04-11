# 데이터셋 변경하여 진행(breast-cancer dataset)
# tensorflow version : 2.12.0
# 실행 명령어 : python 10_dynamic+zero+knn.py --seed 0 --missing_rate 20 --num_mi 5 --m 10 --tau 0.05
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from setproctitle import *
setproctitle('hyejin')
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from dynamic_imputation_model import Dynamic_imputation_nn
from dynamic_imputation_preprocessing import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import argparse
from math import sqrt
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/res/RMSE/16_rice_ensemble_method_res.csv'

# 결과를 저장할 리스트 초기화
results = []

class DynamicImputationModel:
    def __init__(self, num_layers, num_hidden, dim_y, num_features):
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.dim_y = dim_y
        self.num_features = num_features
        tf.compat.v1.disable_eager_execution()
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, self.num_features])

        self.y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, dim_y])
        self.logits, self.pred = self.build_model(self.x)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_true, logits=self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = self.optimizer.minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, x):
        for _ in range(self.num_layers):
            x = tf.layers.dense(x, self.num_hidden, activation=tf.nn.tanh)
        logits = tf.layers.dense(x, self.dim_y)

        if self.dim_y == 1:
            pred = tf.nn.sigmoid(logits)
        elif self.dim_y > 2:
            pred = tf.nn.softmax(logits)

        return logits, pred

    def train_model(self, train_X, train_y, num_epochs, batch_size):
        num_batches = int(np.ceil(len(train_X) / batch_size))
        for epoch in range(num_epochs):
            indices = np.arange(len(train_X))
            np.random.shuffle(indices)
            train_X_shuffled = train_X.iloc[indices]
            train_y_shuffled = train_y.iloc[indices]

            for i in range(num_batches):
                batch_X = train_X_shuffled.iloc[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y_shuffled.iloc[i * batch_size: (i + 1) * batch_size]

                self.sess.run(self.train_op, feed_dict={self.x: batch_X.values, self.y_true: batch_y.values.reshape(-1, 1)})

    def get_accuracy(self, x_tst, y_tst):
        if self.dim_y == 1:
            pred_Y = tf.cast(self.pred > 0.5, tf.float32)
            correct_prediction = tf.equal(pred_Y, self.y_true)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            acc = self.sess.run(accuracy, feed_dict={self.x: x_tst, self.y_true: y_tst})

        else:
            y_tst_hat = self.sess.run(self.pred, feed_dict={self.x: x_tst})
            y_tst_hat = np.argmax(y_tst_hat, axis=1)

            acc = accuracy_score(np.argmax(y_tst, axis=1), y_tst_hat)

        return acc

accuracy_list = []
rmse_list = []

def main(args):

    seed = args.seed
    missing_rate = args.missing_rate
    
    hyperparameters = {'num_mi': args.num_mi, 'm': args.m, 'tau': args.tau}

    prepro_data = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing/16_rice.csv'
    prepro_data = pd.read_csv(prepro_data)

    data_pth = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/missing/16_rice.csv'
    df_data = pd.read_csv(data_pth)
    train_col = ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 'Eccentricity', 'Convex_Area', 'Extent']
    prepro_x = prepro_data[train_col]
    prepro_y = prepro_data['class']

    data = df_data
    
    x = data[train_col].values
    y = data['class'].values

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

        # zero imputation을 위해 데이터 프레임으로 전환
        x_trnval_zero = pd.DataFrame(x_trnval, columns=train_col)
        y_trnval_zero = pd.DataFrame(y_trnval, columns=['class'])
        x_tst_zero = pd.DataFrame(x_tst, columns=train_col)
        y_tst_zero = pd.DataFrame(y_tst, columns=['class'])
        

        # knn imputation을 위해 데이터 프레임으로 전환
        x_trnval_knn = pd.DataFrame(x_trnval, columns=train_col)
        y_trnval_knn = pd.DataFrame(y_trnval, columns=['class'])
        x_tst_knn = pd.DataFrame(x_tst, columns=train_col)
        y_tst_knn = pd.DataFrame(y_tst, columns=['class'])
        
        # zero imputation
        x_trnval_zero_imputed = x_trnval_zero.fillna(0)
        y_trnval_zero_imputed = y_trnval_zero.fillna(0)
        x_txt_zero_imputed = x_tst_zero.fillna(0)
        y_txt_zero_imputed = y_tst_zero.fillna(0)

        # zero imputation 학습 위한 데이터 준비
        train_X_zero_imputed = x_trnval_zero_imputed
        train_y_zero_imputed = y_trnval_zero_imputed
        test_X_zero_imputed = x_txt_zero_imputed
        test_y_zero_imputed = y_txt_zero_imputed

        # knn imputation
        imputer = KNNImputer(n_neighbors=5)
        train_data_knn_imputed = pd.DataFrame(imputer.fit_transform(x_trnval_knn), columns=train_col)
        test_data_knn_imputed = pd.DataFrame(imputer.transform(x_tst_knn), columns=train_col)

        # knn imputation 학습 위한 데이터 준비
        train_X_knn_imputed = train_data_knn_imputed
        train_y_knn_imputed = y_trnval_knn
        test_X_knn_imputed = test_data_knn_imputed
        test_y_knn_imputed = y_tst_knn

        # 신경망 모델 초기화 및 학습 (Zero Imputation)
        model_zero_imputation = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))
        model_zero_imputation.train_model(train_X_zero_imputed, train_y_zero_imputed, num_epochs=50, batch_size=32)
        accuracy_zero_imputation = model_zero_imputation.get_accuracy(test_X_zero_imputed.values, test_y_zero_imputed.values.reshape(-1, 1))

        # 신경망 모델 초기화 및 학습 (knn Imputation)
        model_knn_imputation = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))
        model_knn_imputation.train_model(train_X_knn_imputed, train_y_knn_imputed, num_epochs=50, batch_size=32)
        accuracy_knn_imputation = model_knn_imputation.get_accuracy(test_X_knn_imputed.values, test_y_knn_imputed.values.reshape(-1, 1))

        # dynamic 신경망 모델
        model = Dynamic_imputation_nn(dim_x, dim_y, seed)
        model.train_with_dynamic_imputation(x_trnval, y_trnval, save_path, **hyperparameters)
        acc = model.get_accuracy(x_tst, y_tst)

        # dynamic imputation 결과
        imputed_train_data = model.impute_data(x_trnval)
        # print("Imputed Data for Experiment {}: {}".format(i+1, imputed_train_data))
        # print(imputed_train_data)
        imputed_test_data = model.impute_data(x_tst)
        # print("Imputed Data for Experiment {}: {}".format(i+1, imputed_test_data))
        # print(imputed_test_data)
        
        # 모델 학습 후 imputation 결과 확인
        knn_imputed_model = model_knn_imputation.sess.run(model_knn_imputation.pred, feed_dict={model_knn_imputation.x: test_X_knn_imputed.values})
        dynamic_imputed_model = model.sess.run(model.pred, feed_dict={model.x: imputed_test_data})
        zero_imputed_model = model_zero_imputation.sess.run(model_zero_imputation.pred, feed_dict={model_zero_imputation.x: test_X_zero_imputed.values})

        # 예측값 평균 계산
        avg_predictions = (knn_imputed_model + dynamic_imputed_model + zero_imputed_model) / 3

        # accuracy 계산
        ensemble_accuracy = accuracy_score(y_tst, np.round(avg_predictions))
        accuracy_list.append(ensemble_accuracy)
        ensemble_accuracy_std = np.std(accuracy_list)

        # 결측치 생성 전의 데이터를 동일하게 train/test로 나누어서 저장
        original_x_train, original_x_test, original_y_train, original_y_test = train_test_split(prepro_x, prepro_y, test_size=0.2, random_state=i)

        # RMSE 계산
        rmse = sqrt(((original_y_test.values - avg_predictions.flatten()) ** 2).mean())

        # RMSE의 표준편차 계산
        rmse_list.append(rmse)
        rmse_std = np.std(rmse_list)

        print("==========================================")
        print(str(i + 1) + "th Prediction Average : ", avg_predictions)
        print(str(i + 1) + "th Ensemble Accuracy : {:.4f} ± {:.4f}".format(ensemble_accuracy, ensemble_accuracy_std))
        print(str(i + 1) + "th Ensemble RMSE : {:.4f} ± {:.4f}".format(rmse, rmse_std))
        print("==========================================")


        # 결과를 딕셔너리로 저장
        result = {
            'Dataset' : '16_rice',
            'method' : '10_dynamic+zero+knn',
            'Experiment': i + 1,
            'Accuracy': "{:.4f} ± {:.4f}".format(np.mean(accuracy_list), np.std(accuracy_list)),
            'RMSE': "{:.4f} ± {:.4f}".format(rmse, rmse_std)
        }
        results.append(result)

    print("==========================================")
    print("=== Accuracy result : {:.4f} ± {:.4f}".format(sum(accuracy_list)/len(accuracy_list), np.std(accuracy_list)))
    print("=== RMSE result : {:.4f} ± {:.4f}".format(sum(rmse_list)/len(rmse_list), np.std(rmse_list)))
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