import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense

def shuffle_unit(x, groups):
    with tf.name_scope('shuffle_unit'):
        n, d = x.shape
        size = list(x.shape)
        x = tf.reshape(x, shape=[tf.shape(x)[0], groups, d // groups])
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.reshape(x, shape=[tf.shape(x)[0], d])
    return x

def fully_connected_bn_relu(x, out_dim, training=True):
    with tf.name_scope(None, 'fully_connected_bn_relu'):
        x = Dense(out_dim, activation=None, use_bias=False)(x)
        x = BatchNormalization()(x, training=training)
        x = tf.nn.relu(x)
    print("Output Shape after fully_connected_bn:", x.shape)

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

        print("=== Output Shape after shufflenet_v2_block : ", x.shape)

        return x

