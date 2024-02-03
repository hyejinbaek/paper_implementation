# 데이터셋 변경하여 진행(breast-cancer dataset)
# tensorflow version : 2.12.0
# 실행 명령어 : python 2_ensemble_zero+dynamic.py --seed 0 --missing_rate 20 --num_mi 5 --m 10 --tau 0.05
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 3'
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
from tensorflow.keras.layers import Input, Embedding, Flatten
from sklearn.preprocessing import LabelEncoder
from math import sqrt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from math import sqrt

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/res/add_rmse/1_breast_ensemble_method_res.csv'

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
            ## 추가 
            # indices = np.random.choice(len(train_y), size=len(train_y), replace=False)
            ## 
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

def main(args):

    seed = args.seed
    missing_rate = args.missing_rate
    
    hyperparameters = {'num_mi': args.num_mi, 'm': args.m, 'tau': args.tau}
    
    # 데이터 파일 경로 설정
    data_pth = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/missing/1_breast.csv'

    # 데이터 불러오기
    df_data = pd.read_csv(data_pth)
    train_col = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']

    prepro_data = '/userHome/userhome2/hyejin/paper_implementation/00_dataset/preprocessing/1_breast.csv'
    prepro_data = pd.read_csv(prepro_data)
    prepro_data.columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
    prepro_x = prepro_data[train_col].values
    prepro_y = prepro_data['Class'].values

    data = df_data
    
    x = data[train_col].values
    y = data['Class'].values

    # for문에서 뺌
    x,y = preprocessing(x, y, missing_rate, seed)

    acc_list, auroc = [], []
    accuracy_ensemble_list =[]
    rmse_list = []

    for i  in range(2):
        x_trnval, x_tst, y_trnval, y_tst = train_test_split(x,y, test_size=0.2, shuffle=True, random_state=i)
        

        dim_x = x_trnval.shape[1]

        if y_trnval.shape[1] > 2:
            dim_y = y_trnval.shape[1]
        else:
            dim_y = 1
        save_path = ('./{0}_{1}_model'.format(seed, missing_rate))

        # zero imputation을 위해 데이터 프레임으로 전환
        x_trnval_zero = pd.DataFrame(x_trnval, columns=train_col)
        y_trnval_zero = pd.DataFrame(y_trnval, columns=['Class'])
        x_tst_zero = pd.DataFrame(x_tst, columns=train_col)
        y_tst_zero = pd.DataFrame(y_tst, columns=['Class'])
        
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

        # 신경망 모델 초기화 및 학습 (Zero Imputation)
        model_zero_imputation = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))
        model_zero_imputation.train_model(train_X_zero_imputed, train_y_zero_imputed, num_epochs=50, batch_size=32)
        accuracy_zero_imputation = model_zero_imputation.get_accuracy(test_X_zero_imputed.values, test_y_zero_imputed.values.reshape(-1, 1))

        # dynamic 신경망 모델
        model = Dynamic_imputation_nn(dim_x, dim_y, seed)
        model.train_with_dynamic_imputation(x_trnval, y_trnval, save_path, **hyperparameters)
        acc = model.get_accuracy(x_tst, y_tst)

        print("==========================================")
        print(str(i+1)+"th dynamic accuracy === : ", acc)
        print(str(i+1)+"th zero accuracy === : ", accuracy_zero_imputation)
        print("==========================================")

        acc_list.append(acc)
        acc_list.append(accuracy_zero_imputation)

        # zero imputation 결과
        zero_imputed_train_data = model.impute_data(train_X_zero_imputed)
        print("zero Imputed Data for Experiment {}: {}".format(i+1, zero_imputed_train_data))
        print(zero_imputed_train_data)
        zero_imputed_test_data = model.impute_data(x_tst)
        print("zero Imputed Data for Experiment {}: {}".format(i+1, zero_imputed_test_data))
        print(zero_imputed_test_data)

        # dynamic imputation 결과
        imputed_train_data = model.impute_data(x_trnval)
        print("Imputed Data for Experiment {}: {}".format(i+1, imputed_train_data))
        print(imputed_train_data)
        imputed_test_data = model.impute_data(x_tst)
        print("Imputed Data for Experiment {}: {}".format(i+1, imputed_test_data))
        print(imputed_test_data)

        # 결측치 생성 전의 데이터를 동일하게 train/test로 나누어서 저장
        original_x_train, original_x_test, original_y_train, original_y_test = train_test_split(prepro_x, prepro_y, test_size=0.2, random_state=i)

        # X와 y를 따로 분리
        # train_X_ensemble = pd.concat([pd.DataFrame(x_trnval, columns=train_col), x_trnval_zero_imputed])
        train_X_ensemble = pd.concat([pd.DataFrame(imputed_train_data), pd.DataFrame(zero_imputed_train_data)])

        # train_y_ensemble = pd.concat([pd.DataFrame(y_trnval), y_trnval_zero_imputed])
        train_y_ensemble = pd.concat([pd.DataFrame(imputed_test_data), pd.DataFrame(zero_imputed_test_data)])

        #####################################
        # 앙상블 모델 NN(ShuffleNet) 추가하기 #
        #####################################
        # 앙상블 모델을 훈련하고 예측
        model_ensemble = shuffleNetModel(num_layers=3, num_hidden=128, dim_y=1, num_features=len(train_col))
        model_ensemble.shuffle_train_model(train_X_ensemble, train_y_ensemble, num_epochs=50, batch_size=32)

        # 앙상블 모델로 테스트 데이터 예측
        test_X_ensemble = pd.concat([x_tst, x_tst_zero])
        test_y_ensemble = pd.concat([y_tst, y_tst_zero])
        predictions_ensemble = model_ensemble.sess.run(model_ensemble.pred, feed_dict={model_ensemble.x: test_X_ensemble.values})

        # 앙상블 모델로 예측한 정확도 출력
        accuracy_ensemble = model_ensemble.get_accuracy(test_X_ensemble.values, test_y_ensemble.values.reshape(-1, 1))
        print("==========================================")
        print(str(i + 1) + "th Ensemble Imputation accuracy: ", accuracy_ensemble)
        print("==========================================")

        accuracy_ensemble_list.append(accuracy_ensemble)

        # 결과를 딕셔너리로 저장
        result = {
            'Dataset' : '1_breast',
            'method' : '2_zero + dynamic',
            'Experiment': i + 1,
            'Accuracy': "{:.4f} ± {:.4f}".format(np.mean(acc_list), np.std(acc_list))
        }
        results.append(result)

        # RMSE 계산을 위해 결측치 생성 전의 데이터로 모델 예측
        predictions_original_data = model_ensemble.sess.run(model_ensemble.pred, feed_dict={model_ensemble.x: original_y_test})
        
        # RMSE 계산 및 리스트에 추가
        rmse = sqrt(mean_squared_error(original_y_test, predictions_original_data))
        print("==========================================")
        print(str(i + 1) + "th Ensemble Imputation rmse: ", rmse)
        print("==========================================")
        rmse_list.append(rmse)


    print("==========================================")
    print("=== Ensemble Accuracy : {:.4f} ± {:.4f}".format(np.mean(acc_list), np.std(acc_list)))
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