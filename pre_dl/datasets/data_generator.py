import os
import sys
import glob
import calendar
import datetime

import h5py
import gdal
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

import warnings
warnings.filterwarnings("ignore")

VALID_RANGE = {
    "H8": [0, 335],
    "loc": [0, 1],
    "pre": [0, 100],
    "dem": [-10, 2965]
}

H8_BANDS = ["EVB0860", "EVB1040", "EVB1230"]


class H5BasicReader(object):

    def __init__(self, file_path):
        self.h5_obj = h5py.File(file_path, 'r')
        self.attributes = {key: self.h5_obj.attrs.get(key) for key in self.h5_obj.attrs.keys()}
        self.dataset = None

        if self.h5_obj.visititems(lambda name, node: isinstance(node, h5py.Group)):
            self.dataset = {g_n: {key: value for key, value in g_d.items()} for g_n, g_d in self.h5_obj.items()}

        if self.h5_obj.visititems(lambda name, node: isinstance(node, h5py.Dataset)):
            self.dataset = {key: value for key, value in self.h5_obj.items()}

        if self.h5_obj.visititems(lambda name, node: isinstance(node, h5py.Empty)):
            print('HDF file is empty!')
            sys.exit(1)

    def read(self, band_name, group_name=None):
        if group_name is not None:
            dataset_obj = self.dataset.get(group_name).get(band_name)
        else:
            dataset_obj = self.dataset.get(band_name)
        slope = dataset_obj.attrs["slope"]
        return dataset_obj[:] * slope


class GeoTIFFReader(object):

    def __init__(self, in_file):
        self.in_file = in_file

        dataset = gdal.Open(self.in_file)
        self.XSize = dataset.RasterXSize
        self.YSize = dataset.RasterYSize
        self.GeoTransform = dataset.GetGeoTransform()
        self.ProjectionInfo = dataset.GetProjection()

    def read(self, ):

        dataset = gdal.Open(self.in_file)
        data = dataset.ReadAsArray(0, 0, self.XSize, self.YSize)
        return data

    def get_lon_lat(self, ):

        gtf = self.GeoTransform
        x_range = range(0, self.XSize)
        y_range = range(0, self.YSize)
        x, y = np.meshgrid(x_range, y_range)
        lon = gtf[0] + x * gtf[1] + y * gtf[2]
        lat = gtf[3] + x * gtf[4] + y * gtf[5]
        return lon, lat


class DataGenerator(Sequence):

    def __init__(self, files_df, parse_func, batch_size, shuffle=True):
        self.files_df = files_df
        self.indexes = np.arange(len(files_df))
        self.parse_func = parse_func
        self.batch_size = batch_size
        self.shuffle = shuffle

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.files_df) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        x_lst, pre_lst, ppre_lst = list(), list(), list()
        for idx in batch_indexes:
            x_lst.append(self.parse_func(self.files_df.iloc[idx, :])[0])
            pre_lst.append(self.parse_func(self.files_df.iloc[idx, :])[1])
            ppre_lst.append(self.parse_func(self.files_df.iloc[idx, :])[2])

        batch_x = np.asarray(x_lst)
        batch_pre = np.asarray(pre_lst)
        batch_ppre = np.asarray(ppre_lst)

        return batch_x, {"ppre": batch_ppre, "pre": batch_pre}


def get_items(file_dir):
    df = pd.DataFrame(
        columns=[
            "t", "t-10_H8", "t-10_pre", "t-10_loc", "t_H8", "t_loc", "t_dem", "t_pre"
        ]
    )

    for f in sorted(glob.glob(os.path.join(file_dir, "H8", "*.HDF"))):
        yyyymmdd, hhmm = os.path.basename(f).split("_")[2:4]
        next_ten_h8 = os.path.join(file_dir, "H8", "Himawari8_OBI_{}_{}_PRJ3.HDF".format(yyyymmdd, int(hhmm)+10))
        if os.path.exists(next_ten_h8):
            new_df = pd.DataFrame(
                {
                    "t": yyyymmdd+str(int(hhmm)+10),
                    "t-10_H8": f,
                    "t-10_pre": os.path.join(file_dir, "Pre", "{}_Pre.tif".format(yyyymmdd+hhmm)),
                    "t-10_loc": os.path.join(file_dir, "Location", "{}_Location.tif".format(yyyymmdd + hhmm)),
                    "t_H8": next_ten_h8,
                    "t_loc": os.path.join(file_dir, "Location", "{}_Location.tif".format(yyyymmdd + str(int(hhmm)+10))),
                    "t_dem": os.path.join(file_dir, "DEM", "DEM_H8.tif"),
                    "t_pre": os.path.join(file_dir, "Pre", "{}_Pre.tif".format(yyyymmdd+str(int(hhmm)+10)))
                },
                index=[0]
            )
            df = pd.concat([df, new_df])
        else:
            continue

    return df


def normalization(x, key):
    return (x - VALID_RANGE.get(key)[0]) / (VALID_RANGE.get(key)[1] - VALID_RANGE.get(key)[0])


def cosine_time_radians(time_str, mode):
    time_obj = datetime.datetime.strptime(time_str, "%Y%m%d%H%M")
    j_day = int(time_obj.strftime("%j"))
    j_hour = int(time_obj.hour)
    j_minute = int(time_obj.minute) // 10
    total_days = 366 if calendar.isleap(int(time_obj.year)) else 365
    if "year_time" == mode:
        time_radians = np.linspace(0, 2*np.pi, total_days*24*6)[(j_day-1)*24*6+(j_hour-1)*6+j_minute]
    elif "day_time" == mode:
        time_radians = np.linspace(0, 2*np.pi, 24*6)[(j_hour-1)*6+j_minute]
    else:
        raise ValueError("The value of mode must be one of year_time and day_time, not others!")

    return np.cos(time_radians)


def parse_func(series):
    x_tmp_lst = list()
    ppre_tmp_lst = list()
    pre_tmp_lst = list()

    past_h5_reader = H5BasicReader(series["t-10_H8"])
    for b_name in H8_BANDS:
        tmp_data = past_h5_reader.read(b_name)
        x_tmp_lst.extend([normalization(tmp_data, "H8")])

    past_pre = GeoTIFFReader(series["t-10_pre"]).read()
    x_tmp_lst.extend([normalization(past_pre, "pre")])

    day_time = np.zeros(past_pre.shape)
    day_time[:, :] = cosine_time_radians(series["t"], "day_time")
    x_tmp_lst.extend([day_time])

    year_time = np.zeros(past_pre.shape)
    year_time[:, :] = cosine_time_radians(series["t"], "year_time")
    x_tmp_lst.extend([year_time])

    x_tmp_lst.extend(
        [
            normalization(GeoTIFFReader(series["t-10_loc"]).read(), "loc")
        ]
    )

    cur_h5_reader = H5BasicReader(series["t_H8"])
    for b_name in H8_BANDS:
        tmp_data = cur_h5_reader.read(b_name)
        x_tmp_lst.extend([normalization(tmp_data, "H8")])
    x_tmp_lst.extend(
        [
            normalization(GeoTIFFReader(series["t_loc"]).read(), "loc")
        ]
    )
    x_tmp_lst.extend(
        [
            normalization(GeoTIFFReader(series["t_dem"]).read(), "dem")
        ]
    )

    pre = GeoTIFFReader(series["t_pre"]).read()
    ppre = np.zeros((2, pre.shape[0], pre.shape[1]))
    pre_tmp_lst.extend([normalization(pre, "pre")])
    ppre[1, :][np.where(pre > 0)] = 1
    ppre_tmp_lst.extend(ppre)

    x_arr = np.asarray(x_tmp_lst)
    x_arr = np.transpose(x_arr, [1, 2, 0])
    pre_arr = np.asarray(pre_tmp_lst)
    pre_arr = np.transpose(pre_arr, [1, 2, 0])
    ppre_arr = np.asarray(ppre_tmp_lst)
    ppre_arr = np.transpose(ppre_arr, [1, 2, 0])

    return x_arr, pre_arr, ppre_arr


def train_test_split(top_data_dir, train_size=0.8):
    files_df = get_items(top_data_dir)
    train_lens = int(np.ceil(len(files_df) * train_size))
    valid_lens = int(np.ceil((len(files_df) - train_lens) / 2))

    train_files_df = files_df.iloc[:train_lens, :]
    valid_files_df = files_df.iloc[train_lens:train_lens+valid_lens, :]
    test_files_df = files_df.iloc[train_lens + valid_lens:, :]

    return train_files_df, valid_files_df, test_files_df


def data_generator(top_data_dir, batch_size, train_size=0.8):
    train_files, valid_files, _ = train_test_split(top_data_dir, train_size)
    train_gen = DataGenerator(train_files, parse_func, batch_size)
    valid_gen = DataGenerator(valid_files, parse_func, batch_size)
    return train_gen, valid_gen


if __name__ == '__main__':
    nn_data_dir = r"/hxqtmp/DPLearning/hm/data/PRE"
    train_gen, valid_gen = data_generator(nn_data_dir, 16)
