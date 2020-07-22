import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K


class PRENet(object):

    def __init__(self):
        pass

    def _conv2d_bn(self, x, filters, num_row, num_col, padding='same', strides=(1, 1), use_bias=False):

        x = layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=use_bias)(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.Activation('relu')(x)
        return x

    def _inception_256(self, x):
        conv1x1 = self._conv2d_bn(x, 64, 1, 1)

        conv1x1_3x3 = self._conv2d_bn(x, 128, 1, 1)
        conv1x1_3x3 = self._conv2d_bn(conv1x1_3x3, 128, 3, 3)

        conv1x1_5x5 = self._conv2d_bn(x, 64, 1, 1)
        conv1x1_5x5 = self._conv2d_bn(conv1x1_5x5, 32, 5, 5)

        conv1x1_7x7 = self._conv2d_bn(x, 32, 1, 1)
        conv1x1_7x7 = self._conv2d_bn(conv1x1_7x7, 32, 7, 7)

        merged_x = layers.concatenate([conv1x1, conv1x1_3x3, conv1x1_5x5, conv1x1_7x7])
        return merged_x

    def _inception_512(self, x):
        conv1x1 = self._conv2d_bn(x, 64, 1, 1)

        conv1x1_3x3 = self._conv2d_bn(x, 256, 1, 1)
        conv1x1_3x3 = self._conv2d_bn(conv1x1_3x3, 384, 3, 3)

        conv1x1_5x5 = self._conv2d_bn(x, 64, 1, 1)
        conv1x1_5x5 = self._conv2d_bn(conv1x1_5x5, 32, 5, 5)

        conv1x1_7x7 = self._conv2d_bn(x, 32, 1, 1)
        conv1x1_7x7 = self._conv2d_bn(conv1x1_7x7, 32, 7, 7)

        merged_x = layers.concatenate([conv1x1, conv1x1_3x3, conv1x1_5x5, conv1x1_7x7])
        return merged_x

    def _residual_module(self, input_tensor, filters):

        filters1, filters2 = filters

        x = self._conv2d_bn(input_tensor, filters1, 1, 1)
        x = self._conv2d_bn(x, filters2, 3, 3)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x

    def _nn(self, input_shape, valid_rain):
        inputs = Input(shape=input_shape, name='inputs')

        x = self._conv2d_bn(inputs, 64, 3, 3)
        x = self._conv2d_bn(x, 64, 3, 3)
        residual_64 = self._residual_module(x, [64, 64])
        x = layers.MaxPooling2D()(x)

        x = self._conv2d_bn(x, 128, 3, 3)
        x = self._conv2d_bn(x, 128, 3, 3)
        residual_128 = self._residual_module(x, [128, 128])
        x = layers.MaxPooling2D()(x)

        x = self._inception_256(x)
        x = self._inception_256(x)
        residual_256 = self._residual_module(x, [256, 256])
        x = layers.MaxPooling2D()(x)

        x = self._inception_512(x)
        x = self._inception_512(x)

        x = layers.Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = layers.concatenate([x, residual_256])
        x = self._inception_256(x)
        x = self._inception_256(x)

        x = layers.Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = layers.concatenate([x, residual_128])
        x = self._conv2d_bn(x, 128, 3, 3)
        x = self._conv2d_bn(x, 128, 3, 3)

        x = layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = layers.concatenate([x, residual_64])
        x_p = self._conv2d_bn(x, 128, 3, 3)
        x_val = self._conv2d_bn(x, 128, 3, 3)
        x_p = layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same')(x_p)
        x_p = layers.Activation('sigmoid', name='ppre')(x_p)
        x_val = layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same')(x_val)
        x_val = layers.Lambda(lambda lx: K.clip(lx, valid_rain[0], valid_rain[1]), name="pre")(x_val)

        model = Model(inputs=inputs, outputs=[x_p, x_val])
        print(model.summary())
        return model

    def compile(self, input_shape, valid_rain, lr, lr_decay):
        model = self._nn(input_shape, valid_rain)

        delta_bce = K.variable(0.5)
        delta_mse = K.variable(0.5)
        last_layer_name = ["ppre", "pre"]
        for name in last_layer_name:
            last_layer = model.get_layer(name)
            if "ppre" == name:
                loss = (1/delta_bce*tf.reduce_mean(last_layer.output, keepdims=True)+K.log(delta_bce))

            if "pre" == name:
                loss = (1/delta_mse*tf.reduce_mean(last_layer.output, keepdims=True)+K.log(delta_mse))
            model.add_loss(loss)

        model.compile(
            optimizer=optimizers.Adam(lr=lr, decay=lr_decay),
            loss=[None] * len(model.outputs)
        )


if __name__ == '__main__':
    PRENet().compile(input_shape=(344, 360, 12), valid_rain=(0, 100), lr=1e-3, lr_decay=1e-4)
