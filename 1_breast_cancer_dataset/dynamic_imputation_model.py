# coding: utf-8
# tensorflow v2.8
# uci - breast cancer dataset

import tensorflow as tf
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
# tensorflow v2.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Dynamic_imputation_nn():
    
    def __init__(self, dim_x, dim_y, seed, num_hidden=50, num_layers=1, lr=1e-3, batch_size=32, max_epochs=50):
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.seed = seed
        
    
        # tensorflow v1.x : tf.reset_default_graph()
        # tensorflow v2.x : tf.compat.v1.reset_default_graph()
        # reset_default_graph() : 매번 모델링을 할 때마다 동일한 결과를 얻으려면 실행해야함.
        tf.compat.v1.reset_default_graph()
        self.G = tf.Graph()
        # as_default() : 기본 그래프로 지정
        self.G.as_default()
        
        # tensorflow v1.x
        #self.x = tf.placeholder(tf.float32, shape=(None, dim_x))
        #self.y = tf.placeholder(tf.float32, shape=(None, dim_y))
        # tensorflow v2.x
        tf.compat.v1.disable_eager_execution() # tf 2버전에서 활성화 필요
        # tf.compat.v1.placeholder() : 처음 변수 선언할 때 값을 바로 주는 것이 아닌 나중에 값을 던져주는 공간을 만들어주는 것이다
        self.x = tf.compat.v1.placeholder(tf.float32, shape=(None, dim_x))
        self.y = tf.compat.v1.placeholder(tf.float32, shape=(None, dim_y))

        self.logits, self.pred = self.prediction(self.x)
        
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        
        # skleanr -> IterativeImputer(회귀 대치)
        self.imputer = IterativeImputer(sample_posterior=True, random_state = self.seed)
        
    # [240115] RMSE 때문에 추가 
    def predict(self, x):
        # 이 메서드는 학습된 모델을 사용하여 입력 데이터에 대한 예측값을 반환합니다.
        # 여기서는 모델의 출력값인 pred를 반환하도록 하겠습니다.
        return self.sess.run(self.pred, feed_dict={self.x: x})
    
    # [240123]dynamic imputation 결과를 확인하기 위해 추가
    def impute_data(self, x):
        # 모델이 학습된 후에 호출되는 메서드로, 입력 데이터의 누락된 값을 대체하여 반환합니다.
        # 여기서는 IterativeImputer를 사용한 예시입니다.
        imputed_data = self.imputer.transform(x)
        return imputed_data
    
    def prediction(self, x):
        with tf.variable_scope('network'):
            for _ in range(self.num_layers):
                x = tf.layers.dense(x, self.num_hidden, activation= tf.nn.tanh)
            logits = tf.layers.dense(x, self.dim_y)
            
            if self.dim_y == 1:
                pred = tf.nn.sigmoid(logits)
            
            elif self.dim_y > 2:
                pred = tf.nn.softmax(logits)
                
        return logits, pred
    
    
    def train_with_dynamic_imputation(self, x_trnval, y_trnval, save_path, num_mi, m, tau, early_stopping=True):
        
        self.imputer.fit(x_trnval)
        
        x_trn, x_val, y_trn, y_val = train_test_split(x_trnval, y_trnval, random_state=self.seed, test_size=0.2)
        
        x_val_imputed_list = [self.imputer.transform(x_val) for _ in range(num_mi)]
        x_val_imputed = np.mean(x_val_imputed_list, 0)
        
        n_batch = int(len(x_trn)/self.batch_size)


        if self.dim_y == 1:
            # sigmoid_cross_entropy_with_logits, softmax_cross_entropy_with_logits : 손실함수
            # 차원에 따라 sigmoid/softmax 방법을 각각 적용하여 진행
            # reduce_mean : 텐서플로우 차원을 줄이면서 연산하는 함수로, 특정 차원을 제거하고 평균을 구한다.
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.y, logits = self.logits))
        elif self.dim_y > 2: 
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = self.logits))
            
        # Optimizer : 손실 함수를 통해 얻은 손실값으로 모델을 업데이트하는 방식
        train_op = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(cost)
        
        self.sess.run(tf.global_variables_initializer())
        
        print('::::: training')
        
        val_log = np.zeros(self.max_epochs)
        
        imputed_list= []
        
        for epoch in range(self.max_epochs):
            
            x_trn_imputed = self.imputer.transform(x_trn)
            imputed_list.append(x_trn_imputed)
                    
            [x_trn_input, y_trn_input] = self._permutation([x_trn_imputed, y_trn])
  
            for i in range(n_batch):
                start_ = i*self.batch_size
                end_ = start_ + self.batch_size
                assert self.batch_size == end_ - start_
            
                self.sess.run(train_op, feed_dict={self.x: x_trn_input[start_:end_], self.y:y_trn_input[start_:end_]})
            
            val_loss = self.sess.run(cost, feed_dict={self.x:x_val_imputed, self.y:y_val})
            val_log[epoch] = val_loss
            print('epoch: %d, val_loss: %f, BEST: %f'%(epoch+1, val_loss, np.min(val_log[:epoch+1])))
            
            
        
    def get_accuracy(self, x_tst, y_tst):
        # ±        
        if self.dim_y == 1:
            # tf.cast : 텐서를 새로운 형태로 캐스팅 하는 데 사용
            pred_Y = tf.cast(self.pred > 0.5, tf.float32)
            correct_prediction = tf.equal(pred_Y, self.y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            acc = self.sess.run(accuracy, feed_dict= {self.x: x_tst, self.y: y_tst})
                    
        else:
            y_tst_hat = self.sess.run(self.pred, feed_dict={self.x: x_tst})
            # np.argmax : 함수 내에 array와 비슷한 형태(리스트 등)의 input을 넣어주면 가장 큰 원소의 인덱스 반환
            y_tst_hat = np.argmax(y_tst_hat, axis=1)
        
            acc = accuracy_score(np.argmax(y_tst, axis=1), y_tst_hat)
        
        return acc
    
   
    def get_auroc(self, x_tst, y_tst):
        
        y_tst_hat = self.sess.run(self.pred, feed_dict={self.x: x_tst})
        if self.dim_y == 1:
            auroc = roc_auc_score(y_tst, y_tst_hat)
            
        else:
            auroc = roc_auc_score(y_tst, y_tst_hat, average = 'macro', multi_class = 'ovr')
            
        return auroc

    
    def _permutation(self, set):
        
        permid = np.random.permutation(len(set[0]))
        for i in range(len(set)):
            set[i] = set[i][permid]
        
        return set