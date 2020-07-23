from pre_dl.datasets.data_generator import data_generator
from pre_dl.model.PRENet import PRENet

import os
import re
import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import backend as KB
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, TensorBoard

import warnings

warnings.filterwarnings("ignore")

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 

def main():
    # 参数
    bs = 2
    end_epoch = 100
    nn_data_dir = r"/hxqtmp/DPLearning/hm/data/PRE"
    # 数据批
    train_gen, valid_gen = data_generator(nn_data_dir, bs)
    np.random.seed(620)

    # # 模型准备
    model = PRENet().compile(main_input_shape=(344, 360, 12), gt_ppre_input_shape=(344, 360, 1),gt_pre_input_shape=(344, 360, 1), valid_rain=(0, 100), lr=1e-3, lr_decay=1e-4)
    
    # # 训练模型
    log_dir = './data/round1'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    mcp_save_bset = ModelCheckpoint(os.path.join(log_dir, 'cnns_best.h5'), save_best_only=True, monitor='loss', mode='min')
    mcp_save = ModelCheckpoint(os.path.join(log_dir, 'cnns_{epoch}.h5'), period=1)
    train_logger = CSVLogger(os.path.join(log_dir, 'train_log.csv'))
    events_dir = os.path.join(log_dir, 'events')
    tensorboard = TensorBoard(
        log_dir=events_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=True
    )
    
    history = model.fit(x=train_gen,
                        batch_size=bs,
                        validation_data=valid_gen,
                        epochs=end_epoch,
                        callbacks=[
                            mcp_save,
                            mcp_save_bset,
                            train_logger,
                            tensorboard
                        ], 
                        )
    
if __name__ == '__main__':
    main()