# 데이터셋 변경하여 진행(breast-cancer dataset)
# tensorflow version : 2.12.0
# 실행 명령어 : python test.py --seed 0 --missing_rate 20 --num_mi 5 --m 10 --tau 0.05
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
from tensorflow.keras.layers import Input, Embedding, Flatten
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score

# CSV 파일 경로 설정
result_csv_path = '/userHome/userhome2/hyejin/paper_implementation/experiment_result.csv'

# 결과를 저장할 리스트 초기화
results = []

class DynamicImputationModel:
    def __init__(self, num_layers, num_hidden, dim_y, train_X, train_y):
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.dim_y = dim_y
        tf.compat.v1.disable_eager_execution()
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, train_X.shape[1]])
        self.y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.logits, self.pred = self.build_model(self.x)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_true, logits=self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = self.optimizer.minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        self.train_X = train_X  # Store train_X and train_y as instance variables
        self.train_y = train_y

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
accuracy_list = []
def main(args):

    seed = args.seed
    #dataset = args.dataset
    missing_rate = args.missing_rate
    
    hyperparameters = {'num_mi': args.num_mi, 'm': args.m, 'tau': args.tau}

    data_pth = './breast-cancer.data'
    df_data = pd.read_csv(data_pth)
    col_data = df_data.columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
    train_col = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
    df_data['node-caps'] = df_data['node-caps'].replace('?',0).astype(str)
    df_data['breast-quad'] = df_data['breast-quad'].replace('?',0).astype(str)
    df_data['Class'] = df_data['Class'].replace({2:0, 4:1})
    # print("==== data === ", df_data)
    
    # 범주형 피처 선택
    categorical_columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'breast', 'breast-quad', 'irradiat']

    # 레이블 인코딩 적용
    df_encoded = label_encode(df_data, categorical_columns)
    # print('========== df_encoded ==========', df_encoded.shape)

    data = df_encoded
    print("==== data === ", data.shape)
    
    # 고정 !!
    if len(data)>10000:
        np.random.seed(seed)
        random_sampled_idx = np.random.choice(len(data), 10000, replace=False)
        data = data[random_sampled_idx]
    
    x = data[train_col].values
    y = data['Class'].values


    # for문에서 뺌
    x,y = preprocessing(x, y, missing_rate, seed)

    acc_list, auroc = [], []
    
    # rmse 추가!!!!
    rmse_list = []

    for i  in range(10):
        x_trnval, x_tst, y_trnval, y_tst = train_test_split(x,y, test_size=0.2, shuffle=True, random_state=i)
        # print("=== x_trnval=== ", x_trnval)
        # print("=== x_tst=== ", x_tst)
        # print("=== y_trnval=== ", y_trnval)
        # print("=== y_tst=== ", y_tst)

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
        model_zero_imputation = DynamicImputationModel(num_layers=3, num_hidden=128, dim_y=1, train_X=train_X_zero_imputed, train_y=train_y_zero_imputed)
        model_zero_imputation.train_model(train_X_zero_imputed, train_y_zero_imputed, num_epochs=50, batch_size=32)
        accuracy_zero_imputation = model_zero_imputation.get_accuracy(test_X_zero_imputed.values, test_y_zero_imputed.values.reshape(-1, 1))
        y_zero_pred = model_zero_imputation.sess.run(model_zero_imputation.pred, feed_dict={model_zero_imputation.x: test_X_zero_imputed.values})
        zero_rmse = sqrt(mean_squared_error(test_y_zero_imputed, y_zero_pred))

        model = Dynamic_imputation_nn(dim_x, dim_y, seed)
        model.train_with_dynamic_imputation(x_trnval, y_trnval, save_path, **hyperparameters)

        # dynamic x_tst_imputed : 테스트 세트에 대한 imputation 수행
        x_tst_imputed = model.imputer.transform(x_tst)
        y_pred = model.sess.run(model.pred, feed_dict={model.x: x_tst_imputed})
        acc = model.get_accuracy(x_tst, y_tst)
        dynamic_rmse = sqrt(mean_squared_error(y_tst, y_pred))

        # Ensemble을 위해 두 모델의 예측을 결합
        combined_predictions = (model.sess.run(model.pred, feed_dict={model.x: x_tst_imputed}) 
                        + model_zero_imputation.sess.run(
                            model_zero_imputation.pred, feed_dict={model_zero_imputation.x: test_X_zero_imputed})) / 2
        
        # rmse 2개의 imputation method RMSE 계산
        rmse_combined = np.sqrt(mean_squared_error(y_pred, combined_predictions))
        #print(" === rmse_combined === ", rmse_combined)
        rmse_list.append(rmse_combined) 
        
        print("==========================================")
        print(str(i+1)+"th dynamic accuracy === : ", acc)
        print(str(i+1)+"th zero accuracy === : ", accuracy_zero_imputation)
        print(str(i+1)+"th dynamic rmse === : ", dynamic_rmse)
        print(str(i+1)+"th zero rmse === : ", np.mean(zero_rmse))
        print(str(i + 1) + "th Ensemble RMSE: {:.4f}".format(rmse_combined))
        print("==========================================")

        acc_list.append(acc)
        

        # 결과를 딕셔너리로 저장
        result = {
            'Dataset' : '1_breast',
            'method' : '2_zero + dynamic',
            'Experiment': i + 1,
            'Accuracy': "{:.4f} ± {:.4f}".format(acc, np.std(acc)),
            'RMSE': "{:.4f} ± {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)),
        }
        results.append(result)


    print("==========================================")
    print("=== result : {:.4f} ± {:.4f}".format(sum(acc_list)/len(acc_list), np.std(acc_list)))
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

    # python main.py --seed 0 --dataset avila --missing_rate 30 --num_mi 5 --m 10 --tau 0.05
    # python dynamic_imputation_main.py --seed 0 --missing_rate 30 --num_mi 5 --m 10 --tau 0.05
    arg_parser = argparse.ArgumentParser(description='Dynamic imputation')
    
    arg_parser.add_argument('--seed', help='Random seed', default=27407, type= int)
    #arg_parser.add_argument('--dataset', help='Dataset name', choices=['avila', 'letter'], default=256, type=str)
    arg_parser.add_argument('--missing_rate', help='Missing rate of dataset', default=20, type=float)
    arg_parser.add_argument('--num_mi', help='Number of multiple imputation for validation set', default=5, type=int)
    arg_parser.add_argument('--m', help='Number of imputations to calculate imputation uncertainty', default=10, type=int)
    arg_parser.add_argument('--tau', help='Threshold of imputation uncertainty', default=0.05, type=float)
    
    args = arg_parser.parse_args()
 
    main(args)