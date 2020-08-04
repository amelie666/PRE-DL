'''
Descripttion: 
version: 
Company: http://www.shinetek.com.cn/
Author: bupengju
Date: 2020-08-04 13:49:49
LastEditors: bupengju
LastEditTime: 2020-08-04 15:19:15
'''
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K


class PRECNN(object):
    def __init__(self):
        pass

    def _conv2d_bn(self, x, filters, num_row, num_col, padding='same', strides=(1, 1), use_bias=False):

        x = layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=use_bias)(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.Activation('relu')(x)
        return x

    def _build(self, main_input_shape, valid_rain):
        valid_rain = np.asarray(valid_rain).astype(np.float32)
        main_input = Input(shape=main_input_shape, name='main_input')

        x = self._conv2d_bn(main_input, 64, 3, 3)
        x = self._conv2d_bn(x, 128, 3, 3)
        x = self._conv2d_bn(x, 64, 3, 3)

        x = layers.Flatten()(x)

        x = layers.Dense(2048, activation="relu")(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dense(1)(x)
        output = layers.Lambda(lambda x: K.clip(x, valid_rain[0], valid_rain[1]))(x)

        model = Model(inputs=main_input, outputs=output, name="AI-QPE")
        print(model.summary())

        return model

    def compile(self, main_input_shape, valid_rain, lr):
        model = self._build(main_input_shape, valid_rain)

        model.compile(
            optimizer=optimizers.Adam(lr=lr),
            loss="mean_squared_error"
        )

        return model


if __name__ == '__main__':
    import os
    # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    PRECNN().compile(
        main_input_shape=(9, 9, 9), valid_rain=(0., 100.), lr=1e-3
    )