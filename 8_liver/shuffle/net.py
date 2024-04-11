# net.py
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
    def __init__(self, cls, model_scale=1.0, shuffle_group=2):
        self.cls = cls
        self.shuffle_group = shuffle_group
        self.fc_dims = self._select_fc_dim(model_scale)

    def _select_fc_dim(self, model_scale):
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

        for idx, fc_dim in enumerate(self.fc_dims):
            with tf.name_scope(f'shuffle_fc_block_{idx}'):

                out = shufflenet_v2_block(out, fc_dim, groups=self.shuffle_group)

        with tf.name_scope('prediction'):
            # 직접 fully_connected_bn 함수 호출

            out = fully_connected_bn(out, self.cls, training)
        return out

    def __call__(self, input_tensor, training=True):
        return self._build_model(input_tensor, training)
