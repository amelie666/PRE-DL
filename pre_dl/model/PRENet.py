import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K


class QPELoss(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(QPELoss, self).__init__(**kwargs)

    def build(self, input_shape):

        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.delta_bce = self.add_weight(name='kernel',
                                         # shape=(input_shape[0].as_list()[1], self.output_dim),
                                         shape=(1, ),
                                         initializer=initializers.constant(1.),
                                         trainable=True)

        self.delta_mse = self.add_weight(name='kernel',
                                         # shape=(input_shape[0].as_list()[1], self.output_dim),
                                         shape=(1, ),
                                         initializer=initializers.constant(1.),
                                         trainable=True)

        super(QPELoss, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x, **kwargs):
        assert isinstance(x, list)
        a, b, c, d = x

        loss = K.mean(K.binary_crossentropy(a, b))/K.square(self.delta_bce)+K.abs(K.log(self.delta_bce))
        loss += K.mean(K.square(c - d))/(2*K.square(self.delta_mse))+K.abs(K.log(self.delta_mse))

        self.add_loss(tf.reduce_mean(loss))

        return tf.reduce_mean(loss)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b, shape_c, shape_d = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]


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

    def _build(self, main_input_shape, gt_ppre_input_shape, gt_pre_input_shape, valid_rain):
        valid_rain = np.asarray(valid_rain).astype(np.float32)
        main_input = Input(shape=main_input_shape, name='main_input')
        gt_ppre = Input(shape=gt_ppre_input_shape, name='gt_ppre')
        gt_pre = Input(shape=gt_pre_input_shape, name='gt_pre')

        x = self._conv2d_bn(main_input, 64, 3, 3)
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

        qpe_loss = QPELoss(1, name="qpe_loss")(
            [
                layers.Flatten()(gt_ppre),
                layers.Flatten()(x_p),
                layers.Flatten()(gt_pre),
                layers.Flatten()(x_val)
            ]
        )

        model = Model(inputs=[main_input, gt_ppre, gt_pre], outputs=[x_p, x_val, qpe_loss], name="AI-QPE")
        print(model.summary())

        return model

    def compile(self, main_input_shape, gt_ppre_input_shape, gt_pre_input_shape, valid_rain, lr, lr_decay):
        model = self._build(main_input_shape, gt_ppre_input_shape, gt_pre_input_shape, valid_rain)

        model.compile(
            optimizer=optimizers.Adam(lr=lr, decay=lr_decay),
            loss=[None] * len(model.outputs)
        )

        # model.compile(
        #     optimizer=optimizers.Adam(lr=lr, decay=lr_decay),
        #     loss={"ppre": losses.binary_crossentropy,
        #           "pre": losses.mean_squared_error},
        #     metrics={
        #         "ppre": metrics.binary_accuracy,
        #         "pre": metrics.mean_absolute_error
        #     },
        #     loss_weights=[0.5, 0.5]
        # )
        return model
