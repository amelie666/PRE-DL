import os
import sys
import glob
import calendar
import datetime
import warnings

warnings.filterwarnings("ignore")

import h5py
import gdal
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence


VALID_RANGE = {
    "H8": [0, 335],
    "loc": [0, 1],
    "pre": [0, 100],
    "dem": [-10, 2965]
}

H8_BANDS = ["EVB0860", "EVB1040", "EVB1230"]
RG_BANDS = ["V13392_010", ]

STUDY_AREA_SHAPE = [344, 360]

WINDOW_SIZE = 32


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

    def get_time(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        times_list = list()
        for idx in batch_indexes:
            times_list.append(self.files_df.loc[idx, 't'])
        return times_list

    def get_row(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        row_list = list()
        for idx in batch_indexes:
            row_list.append(self.files_df.loc[idx, ['start_row','end_row']])
        return row_list

    def get_col(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        col_list = list()
        for idx in batch_indexes:
            col_list.append(self.files_df.loc[idx, ['start_col','end_col']])
        return col_list

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

        return {"main_input": batch_x, "gt_ppre": batch_ppre, "gt_pre": batch_pre}, \
               {"ppre": batch_ppre, "pre": batch_pre}


def get_items(file_dir):
    df = pd.DataFrame(
        columns=[
            "t", "t-10_H8", "t-10_pre", "t-10_loc", "t_H8", "t_dem",
            "t_loc", "t_pre", "start_row", "end_row", "start_col", "end_col"
        ]
    )

    for f in sorted(glob.glob(os.path.join(file_dir, "H8", "*.HDF"))):
        yyyymmdd, hhmm = os.path.basename(f).split("_")[2:4]
        next_ten_h8 = os.path.join(file_dir, "H8", "Himawari8_OBI_{}_{}_PRJ3.HDF".format(yyyymmdd, int(hhmm) + 10))
        if os.path.exists(next_ten_h8):
            t_rg = os.path.join(file_dir, "RG", "{}.csv".format(yyyymmdd + str(int(hhmm) + 10)))
            rg_df = pd.read_csv(t_rg)
            rain_df = rg_df[rg_df["V13392_010"] > 0]
            gauges_num = rain_df.index.values.shape[0]
            no_rain_df = rg_df[rg_df["V13392_010"] == 0].sample(gauges_num, random_state=np.random.seed(42))
            target_df = pd.concat([rain_df, no_rain_df])

            for row in target_df.itertuples():
                col = getattr(row, "v_i")
                row = getattr(row, "u_i")

                if (row - int(WINDOW_SIZE / 2)) < 0 or (col - int(WINDOW_SIZE / 2)) < 0:
                    continue

                if (row + int(WINDOW_SIZE / 2)) > STUDY_AREA_SHAPE[0] or (col + int(WINDOW_SIZE / 2)) > \
                    STUDY_AREA_SHAPE[1]:
                    continue

                new_df = pd.DataFrame(
                    {
                        "t": yyyymmdd + str(int(hhmm) + 10),
                        "t-10_H8": f,
                        "t-10_pre": os.path.join(file_dir, "Pre", "{}_Pre.tif".format(yyyymmdd + hhmm)),
                        "t-10_loc": os.path.join(file_dir, "Location", "{}_Location.tif".format(yyyymmdd + hhmm)),
                        "t_H8": next_ten_h8,
                        "t_loc": os.path.join(file_dir, "Location",
                                              "{}_Location.tif".format(yyyymmdd + str(int(hhmm) + 10))),
                        "t_dem": os.path.join(file_dir, "DEM", "DEM_H8.tif"),
                        "t_pre": os.path.join(file_dir, "Pre", "{}_Pre.tif".format(yyyymmdd + str(int(hhmm) + 10))),
                        "start_row": row - int(WINDOW_SIZE / 2),
                        "end_row": row + int(WINDOW_SIZE / 2),
                        "start_col": col - int(WINDOW_SIZE / 2),
                        "end_col": col + int(WINDOW_SIZE / 2)
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
    time_obj = datetime.datetime.strptime(str(time_str), "%Y%m%d%H%M")
    j_day = int(time_obj.strftime("%j"))
    j_hour = int(time_obj.hour)
    j_minute = int(time_obj.minute) // 10
    total_days = 366 if calendar.isleap(int(time_obj.year)) else 365
    if "year_time" == mode:
        time_radians = np.linspace(0, 2 * np.pi, total_days * 24 * 6)[
            (j_day - 1) * 24 * 6 + (j_hour - 1) * 6 + j_minute]
    elif "day_time" == mode:
        time_radians = np.linspace(0, 2 * np.pi, 24 * 6)[(j_hour - 1) * 6 + j_minute]
    else:
        raise ValueError("The value of mode must be one of year_time and day_time, not others!")

    return np.cos(time_radians)


def parse_func(series):
    x_tmp_lst = list()
    ppre_tmp_lst = list()
    pre_tmp_lst = list()

    start_row = series["start_row"]
    end_row = series["end_row"]
    start_col = series["start_col"]
    end_col = series["end_col"]

    past_h5_reader = H5BasicReader(series["t-10_H8"])
    for b_name in H8_BANDS:
        tmp_data = past_h5_reader.read(b_name)[start_row:end_row, start_col:end_col]
        x_tmp_lst.extend([normalization(tmp_data, "H8")])

    past_pre = GeoTIFFReader(series["t-10_pre"]).read()[start_row:end_row, start_col:end_col]
    x_tmp_lst.extend([normalization(past_pre, "pre")])

    day_time = np.zeros(past_pre.shape)
    day_time[:, :] = cosine_time_radians(series["t"], "day_time")
    x_tmp_lst.extend([day_time])

    year_time = np.zeros(past_pre.shape)
    year_time[:, :] = cosine_time_radians(series["t"], "year_time")
    x_tmp_lst.extend([year_time])

    x_tmp_lst.extend(
        [
            normalization(GeoTIFFReader(series["t-10_loc"]).read()[start_row:end_row, start_col:end_col], "loc")
        ]
    )

    cur_h5_reader = H5BasicReader(series["t_H8"])
    for b_name in H8_BANDS:
        tmp_data = cur_h5_reader.read(b_name)[start_row:end_row, start_col:end_col]
        x_tmp_lst.extend([normalization(tmp_data, "H8")])

    x_tmp_lst.extend(
        [
            normalization(GeoTIFFReader(series["t_dem"]).read()[start_row:end_row, start_col:end_col], "dem")
        ]
    )

    pre = GeoTIFFReader(series["t_pre"]).read()[start_row:end_row, start_col:end_col]
    ppre = np.zeros((pre.shape[0], pre.shape[1]))
    pre_tmp_lst.extend([normalization(pre, "pre")])
    ppre[np.where(pre > 0)] = 1
    ppre_tmp_lst.extend([ppre])

    x_arr = np.asarray(x_tmp_lst)
    x_arr = np.transpose(x_arr, [1, 2, 0])
    pre_arr = np.asarray(pre_tmp_lst)
    pre_arr = np.transpose(pre_arr, [1, 2, 0])
    ppre_arr = np.asarray(ppre_tmp_lst)
    ppre_arr = np.transpose(ppre_arr, [1, 2, 0])

    return x_arr, pre_arr, ppre_arr


def train_test_split(files_df_path, top_data_dir, train_size=0.8):
    if os.path.exists(files_df_path):
        files_df = pd.read_csv(files_df_path)
    else:
        files_df = get_items(top_data_dir)

    unique_time = files_df["t"].unique()
    lens = len(unique_time)
    train_lens = int(np.ceil(lens * train_size))
    valid_lens = int(np.ceil((lens - train_lens) / 2))

    train_time = unique_time[:train_lens]
    valid_time = unique_time[train_lens:train_lens + valid_lens]
    test_time = unique_time[train_lens + valid_lens:]

    train_files_df = files_df[files_df["t"].isin(train_time)]
    valid_files_df = files_df[files_df["t"].isin(valid_time)]
    test_files_df = files_df[files_df["t"].isin(test_time)]

    test_files_df = convert_test_df_for_demo(test_files_df)

    return train_files_df, valid_files_df, test_files_df


def convert_test_df_for_demo(test_files_df):
    p_step = (32, 32)
    unique_time = test_files_df["t"].unique()
    tdf = pd.DataFrame(
        columns=[
            "t", "t-10_H8", "t-10_pre", "t-10_loc", "t_H8", "t_dem",
            "t_loc", "t_pre", "start_row", "end_row", "start_col", "end_col"
        ]
    )

    r = np.arange(0, STUDY_AREA_SHAPE[0], WINDOW_SIZE)
    r = np.clip(r, 0, STUDY_AREA_SHAPE[0] - WINDOW_SIZE)
    c = np.arange(0, STUDY_AREA_SHAPE[1], WINDOW_SIZE)
    c = np.clip(c, 0, STUDY_AREA_SHAPE[1] - WINDOW_SIZE)
    r, c = np.meshgrid(r, c)
    r_count = r.ravel().shape[0]
    for time in unique_time:
        new_tdf = test_files_df[test_files_df['t'] == time].iloc[0:1, :]
        new_tdf = pd.concat([new_tdf] * r_count, ignore_index=True)
        new_tdf.loc[:, 'start_row'] = r.ravel()
        new_tdf.loc[:, 'end_row'] = r.ravel() + WINDOW_SIZE
        new_tdf.loc[:, 'start_col'] = c.ravel()
        new_tdf.loc[:, 'end_col'] = c.ravel() + WINDOW_SIZE
        tdf = pd.concat([tdf, new_tdf], ignore_index=True)
    return tdf


def data_generator(files_df_path, top_data_dir, batch_size, train_size=0.8, train=True):
    train_files, valid_files, test_files = train_test_split(files_df_path, top_data_dir, train_size)
    if train:
        train_gen = DataGenerator(train_files, parse_func, batch_size)
        valid_gen = DataGenerator(valid_files, parse_func, batch_size)
        return train_gen, valid_gen
    else:
        test_gen = DataGenerator(test_files, parse_func, batch_size)
        return test_gen
