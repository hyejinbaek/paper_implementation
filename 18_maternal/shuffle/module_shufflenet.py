import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense

# def shuffle_unit(x, groups):
#     with tf.name_scope('shuffle_unit'):
#         print(" === shuffle_unit_x ===", x.shape)
#         n, d = x.shape
#         print(" === shuffle_unit_n ===", n)
#         print(" === shuffle_unit_d ===", d)
#         size = list(x.shape)
#         print(" === shuffle_unit_size ===", size)
#         x = tf.reshape(x, shape=[tf.shape(x)[0], groups, d // groups])
#         print(" ====shufflenet_unit_x1", x.shape)
#         x = tf.transpose(x, perm=[0, 2, 1])
#         print(" ====shufflenet_unit_x2", x.shape)
#         x = tf.reshape(x, shape=[tf.shape(x)[0], d])
#         print(" ====shufflenet_unit_x3", x.shape)
#     return x

def shuffle_unit(x, groups):
    with tf.name_scope('shuffle_unit'):
        n, d = x.shape
        remainder = d % (groups * 2)  # 텐서 크기가 2의 배수로 나누어 떨어지지 않으면 나머지를 계산
        if remainder != 0:
            pad_size = (groups * 2) - remainder  # 나머지가 0이 될 때까지 텐서를 패딩합니다.
            x = tf.pad(x, [[0, 0], [0, pad_size]])  # 패딩 추가
            d += pad_size  # 패딩된 텐서 크기 업데이트
        group_size = d // groups
        x = tf.reshape(x, shape=[-1, groups, group_size])
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.reshape(x, shape=[-1, d])
    return x




def fully_connected_bn_relu(x, out_dim, training=True):
    with tf.name_scope(None, 'fully_connected_bn_relu'):
        x = Dense(out_dim, activation=None, use_bias=False)(x)
        x = BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
    return x

class FullyConnectedBNLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim, **kwargs):
        super(FullyConnectedBNLayer, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.dense = Dense(out_dim, activation=None, use_bias=False)
        self.batch_norm = BatchNormalization()

    def call(self, x, training=True):
        x = self.dense(x)
        x = self.batch_norm(x, training=training)
        return x

def fully_connected_bn_relu(x, out_dim, training=True):
    with tf.name_scope('fully_connected_bn_relu'):
        x = tf.keras.layers.Dense(out_dim, activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
    return x


def shufflenet_v2_block(x, out_dim, groups=2):
    with tf.name_scope('shuffle_v2_block'):
        n, d = x.shape
        size = list(x.shape)

        # ShuffleNetV2 block
        with tf.name_scope('shuffle_unit'):
            x = shuffle_unit(x, groups)

        # Fully Connected Layer
        with tf.name_scope('fully_connected_bn_relu'):
            x = fully_connected_bn_relu(x, out_dim)

        return x

