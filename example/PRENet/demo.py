from pre_dl.datasets.data_generator import data_generator

import os
import re
import sys
import glob
from datetime import datetime

import rasterio as rio
import numpy as np

from tensorflow.keras.models import load_model

import tqdm

import warnings

warnings.filterwarnings("ignore")

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 


def evaluate_per(ppre, pre, out, time):
    with rio.open(
            os.path.join(out, '%s_ppre.tif' % time),
            'w',
            driver='GTiff',
            height=ppre.shape[0],
            width=ppre.shape[1],
            count=1,
            dtype=ppre.dtype,
            crs='+proj=aea +lat_1=29.5 +lat_2=45.5 +lon_0=116 +datum=WGS84 +units=m',
            transform=rio.transform.Affine.from_gdal(-360822.062, 2000, 0.0, 4485438.524, 0.0, -2000),
    ) as dst:
        dst.write(ppre, 1)
    with rio.open(
            os.path.join(out, '%s_pre.tif' % time),
            'w',
            driver='GTiff',
            height=pre.shape[0],
            width=pre.shape[1],
            count=1,
            dtype=pre.dtype,
            crs='+proj=aea +lat_1=29.5 +lat_2=45.5 +lon_0=116 +datum=WGS84 +units=m',
            transform=rio.transform.Affine.from_gdal(-360822.062, 2000, 0.0, 4485438.524, 0.0, -2000),
    ) as dst:
        dst.write(pre, 1)
        
def main():    
    # 参数
    bs = 1
    nn_data_dir = r"/hxqtmp/DPLearning/hm/data/PRE"
    # 数据批
    _, _, test_gen = data_generator(nn_data_dir, bs)
    model_path = './data/round1/cnns_42.h5'
    out = './data/round1/cnns_42'
    os.makedirs(out, exist_ok=True)
    cnn = load_model(model_path)
    for i in tqdm.trange(len(test_gen), desc='processing'):
        time = test_gen.get_time(i)[0]
        x, y = test_gen[i]
        ppre, pre = cnn.predict(x['main_input'])
        ppre, pre = np.squeeze(ppre),  np.squeeze(pre)
        evaluate_per(ppre, pre, out ,time)



if __name__ == "__main__":
    main()