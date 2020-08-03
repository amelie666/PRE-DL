from pre_dl.datasets.data_generator import data_generator, STUDY_AREA_SHAPE
from pre_dl.model.PRENet import PRENet, QPELoss

import os
import re
import sys
import glob
from datetime import datetime

import rasterio as rio
from rasterio.windows import Window
import numpy as np

from tensorflow.keras.models import load_model

import tqdm

import warnings

warnings.filterwarnings("ignore")

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 


def evaluate_per(ppre, pre, out, time, rows, cols):
    ppre_file = os.path.join(out, '%s_ppre.tif' % time)
    mode = 'r+' if os.path.exists(ppre_file) else 'w'
    with rio.open(
            ppre_file,
            mode,
            driver='GTiff',
            height=STUDY_AREA_SHAPE[0],
            width=STUDY_AREA_SHAPE[1],
            count=1,
            dtype=ppre.dtype,
            crs='+proj=aea +lat_1=29.5 +lat_2=45.5 +lon_0=116 +datum=WGS84 +units=m',
            transform=rio.transform.Affine.from_gdal(-360822.062, 2000, 0.0, 4485438.524, 0.0, -2000),
    ) as dst:
        dst.write(ppre, window=Window(cols[0], rows[0], cols[1]-cols[0], rows[1]-rows[0]), indexes=1)
    with rio.open(
            os.path.join(out, '%s_pre.tif' % time),
            mode,
            driver='GTiff',
            height=STUDY_AREA_SHAPE[0],
            width=STUDY_AREA_SHAPE[1],
            count=1,
            dtype=pre.dtype,
            crs='+proj=aea +lat_1=29.5 +lat_2=45.5 +lon_0=116 +datum=WGS84 +units=m',
            transform=rio.transform.Affine.from_gdal(-360822.062, 2000, 0.0, 4485438.524, 0.0, -2000),
    ) as dst:
        dst.write(pre, window=Window(cols[0], rows[0], cols[1]-cols[0], rows[1]-rows[0]), indexes=1)
        
def main():    
    # 参数
    bs = 1
    # 数据批
    files_df_path= '/hxqtmp/DPLearning/bupj/PRE_DL/debug/tmp/NCHN_mon_5.csv'
    top_data_dir = "/hxqtmp/DPLearning/hm/data/PRE"
     # 模型准备
    start_point = './data/round3/weights_05_0.019908.h5'
    
    test_gen = data_generator(files_df_path, top_data_dir, bs, train=False)
    
    model = load_model(start_point, custom_objects={'QPELoss': QPELoss(1, name="qpe_loss")})
    out = './data/round3/cnns_1'
    os.makedirs(out, exist_ok=True)

    for i in tqdm.trange(len(test_gen), desc='processing'):
        time = test_gen.get_time(i)[0]
        rows = test_gen.get_row(i)[0]
        cols = test_gen.get_col(i)[0]
        x, y = test_gen[i]
        ppre, pre, qe_loss = model.predict(x)
        ppre, pre = np.squeeze(ppre),  np.squeeze(pre)
        evaluate_per(ppre, pre, out ,time, rows, cols)



if __name__ == "__main__":
    main()