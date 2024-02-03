# net.py
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU
from module_shufflenet import shufflenet_v2_block

def fully_connected_bn(x, out_dim, training=True):
    with tf.name_scope('fully_connected_bn'):
        x = Dense(out_dim, activation=None, use_bias=False)(x)
        x = BatchNormalization()(x, training=training)
        x = ReLU()(x)
    return x

class ShuffleNetV2(tf.Module):
    def __init__(self, num_classes=2, shuffle_group=2, model_scale=1.0):
        self.num_classes = num_classes
        self.shuffle_group = shuffle_group
        self.fc_dims = self._select_fc_dim(model_scale)

    def _select_fc_dim(self, model_scale):
        print("==============")
        print("=== model_scale ===", model_scale)
        print("==============")
        if model_scale == 0.5:
            return [48, 96, 192, 1024]
        elif model_scale == 1.0:
            return [116, 232, 464, 1024]
        elif model_scale == 1.5:
            return [176, 352, 704, 1024]
        elif model_scale == 2.0:
            return [244, 488, 976, 2048]
        else:
            raise ValueError('Unsupported model size.')
        
    def _build_model(self, input_tensor, training=True):
        # 가정: 입력 모양은 [배치 크기, 특성 개수]입니다.
        
        out = input_tensor
        print(" ==input_tensor==", input_tensor)
        print(" ==input_tensor==", input_tensor.shape)

        for idx, fc_dim in enumerate(self.fc_dims):
            with tf.name_scope(f'shuffle_fc_block_{idx}'):
                out = shufflenet_v2_block(out, fc_dim, groups=self.shuffle_group)

        with tf.name_scope('prediction'):
            out = fully_connected_bn(out, self.num_classes, training)
        print(" === out === ", out)
        print(" === out === ", out.shape)
        return out

    def __call__(self, input_tensor, training=True):
        return self._build_model(input_tensor, training)

    # def shuffle_train_model(self, train_X_ensemble, train_y_ensemble):
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    #     for epoch in range(50):  # 필요에 따라 epoch 수를 변경하세요
    #         with tf.GradientTape() as tape:
    #             logits_net = self(train_X_ensemble, training=True)
    #             loss_net = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_y_ensemble, logits=logits_net))
    #         gradients_net = tape.gradient(loss_net, self.trainable_variables)
    #         optimizer.apply_gradients(zip(gradients_net, self.trainable_variables))
    #         print(f'Epoch {epoch + 1}, Loss Net: {loss_net.numpy()}')

    def shuffle_train_model(self, train_X_ensemble, train_y_ensemble):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        for epoch in range(50):
            with tf.GradientTape() as tape:
                logits_net = self(train_X_ensemble, training=True)
                loss_net = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_y_ensemble, logits=logits_net))
            gradients_net = tape.gradient(loss_net, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients_net, self.trainable_variables))
            print(f'Epoch {epoch + 1}, Loss Net: {loss_net.numpy()}')
        

    # def get_accuracy(self, x_tst, y_tst):
    #     if self.dim_y == 1:
    #         pred_Y = tf.cast(self.pred > 0.5, tf.float32)
    #         correct_prediction = tf.equal(pred_Y, self.y_true)
    #         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #         acc = self.sess.run(accuracy, feed_dict={self.x: x_tst, self.y_true: y_tst})

    #     else:
    #         y_tst_hat = self.sess.run(self.pred, feed_dict={self.x: x_tst})
    #         y_tst_hat = np.argmax(y_tst_hat, axis=1)

    #         acc = accuracy_score(np.argmax(y_tst, axis=1), y_tst_hat)

    #     return acc
    
    def get_accuracy(self, x_tst, y_tst):
        if self.num_classes == 1:
            pred_Y = tf.cast(self.pred > 0.5, tf.float32)
            correct_prediction = tf.equal(pred_Y, y_tst)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            acc = accuracy.numpy()
        else:
            y_tst_hat = tf.argmax(self.pred, axis=1)
            acc = accuracy_score(y_tst, y_tst_hat)
        return acc

       


