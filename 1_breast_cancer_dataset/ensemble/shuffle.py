# coding: utf-8
# tensorflow v2.8
# uci - breast cancer dataset

import tensorflow as tf
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
# tensorflow v2.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
    
# class ShuffleModel:
#     def __init__(self, nc_input, n_classes):
#         n_fs = 128
#         n_groups = 32

#         first_channels = 32

#         self.stem = tf.keras.layers.Conv1D(first_channels, kernel_size=3, strides=1, padding='same', use_bias=True)
#         self.bn = tf.keras.layers.BatchNormalization()

#         self.stem = tf.keras.layers.Conv1D(first_channels, kernel_size=3, strides=1, padding='same', use_bias=True)
#         self.bn = tf.keras.layers.BatchNormalization()

#         self.block1 = Block(first_channels, n_fs, n_groups, first_grouped_conv=False, pool=True)
#         self.block2 = Block(n_fs, n_fs * 2, n_groups, pool=True)
#         self.block3 = Block(n_fs * 2, n_fs * 2, n_groups)
#         self.final = tf.keras.layers.Dense(n_classes)

#     def Block():
        



#     @tf.function
#     def __call__(self, x):
#         x = self.bn(self.stem(x))
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = tf.nn.adaptive_avg_pool1d(x, 1)
#         x = tf.reshape(x, [tf.shape(x)[0], -1])
#         x = self.final(x)
#         return x

#     def build_model(self, x):
#         for _ in range(self.num_layers):
#             x = tf.layers.dense(x, self.num_hidden, activation=tf.nn.tanh)
#         logits = tf.layers.dense(x, self.dim_y)

#         if self.dim_y == 1:
#             pred = tf.nn.sigmoid(logits)
#         elif self.dim_y > 2:
#             pred = tf.nn.softmax(logits)

#         return logits, pred

#     def train_model(self, train_X, train_y, num_epochs, batch_size):
#         num_batches = int(np.ceil(len(train_X) / batch_size))
#         for epoch in range(num_epochs):
#             indices = np.arange(len(train_X))
#             np.random.shuffle(indices)
#             train_X_shuffled = train_X.iloc[indices]
#             train_y_shuffled = train_y.iloc[indices]

#             for i in range(num_batches):
#                 batch_X = train_X_shuffled.iloc[i * batch_size: (i + 1) * batch_size]
#                 batch_y = train_y_shuffled.iloc[i * batch_size: (i + 1) * batch_size]

#                 self.sess.run(self.train_op, feed_dict={self.x: batch_X.values, self.y_true: batch_y.values.reshape(-1, 1)})

#     def get_accuracy(self, x_tst, y_tst):
#         if self.dim_y == 1:
#             pred_Y = tf.cast(self.pred > 0.5, tf.float32)
#             correct_prediction = tf.equal(pred_Y, self.y_true)
#             accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#             acc = self.sess.run(accuracy, feed_dict={self.x: x_tst, self.y_true: y_tst})

#         else:
#             y_tst_hat = self.sess.run(self.pred, feed_dict={self.x: x_tst})
#             y_tst_hat = np.argmax(y_tst_hat, axis=1)

#             acc = accuracy_score(np.argmax(y_tst, axis=1), y_tst_hat)

#         return acc

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

class ShuffleModel(tf.keras.Model):
    def __init__(self):
        super(ShuffleModel, self).__init__()

        # 첫 번째 입력층 처리
        self.x1 = Dense(64, activation='relu')
        self.x2 = Dense(32, activation='relu')

        # 두 번째 입력층 처리
        self.y1 = Dense(64, activation='relu')
        self.y2 = Dense(32, activation='relu')

        # 출력층
        self.output_layer = Dense(1, activation='sigmoid', name='output')

    def call(self, inputs):
        # 첫 번째 입력층 처리
        x1 = self.x1(inputs[0])
        x1 = self.x2(x1)

        # 두 번째 입력층 처리
        y1 = self.y1(inputs[1])
        y1 = self.y2(y1)

        # 두 입력층 결합
        combined = Concatenate()([x1, y1])

        # 출력층
        output = self.output_layer(combined)

        return output

    def loss(self, y_true, y_pred):
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)

    def accuracy(self, y_true, y_pred):
        return tf.keras.metrics.binary_accuracy(y_true, y_pred)

    def rmse(self, y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

