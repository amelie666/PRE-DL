import os
import sys
import glob
import calendar
import datetime
import warnings
from collections import defaultdict

import h5py
import gdal
import tqdm
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

warnings.filterwarnings("ignore")

VALID_RANGE = {
    "H8": [0, 335],
    "loc": [0, 1],
    "pre": [0, 100],
    "dem": [-10, 2965]
}

H8_BANDS = ["EVB0860", "EVB1040", "EVB1230"]
RG_BANDS = ["V13392_010", ]

STUDY_AREA_SHAPE = [344, 360]

WINDOW_SIZE = 9


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

    def __init__(self, files_df, parse_func, batch_size, shuffle=True, data_list_preprocess=None, **kwarg):
        self.files_df = files_df
        self.indexes = np.arange(len(files_df))
        self.parse_func = parse_func
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_list_preprocess = data_list_preprocess
        self.kwarg = kwarg

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
            row_list.append(self.files_df.loc[idx, ['start_row', 'end_row']])
        return row_list

    def get_col(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        col_list = list()
        for idx in batch_indexes:
            col_list.append(self.files_df.loc[idx, ['start_col', 'end_col']])
        return col_list

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        data_list = []
        for idx in batch_indexes:
            data = self.parse_func(self.files_df.iloc[idx, :], **self.kwarg)
            data_list.append(data)
        if self.data_list_preprocess:
            out_put = self.data_list_preprocess(data_list)
            return out_put
        else:
            return data_list


def get_items(file_dir):
    df = pd.DataFrame(
        columns=[
            "t", "t-10_H8", "t-10_pre", "t-10_loc", "t_H8", "t_dem",
            "t_loc", "t_pre", "start_row", "end_row", "start_col", "end_col"
        ]
    )

    for f in tqdm.tqdm(sorted(glob.glob(os.path.join(file_dir, "H8", "*.HDF"))), desc='Generate DataFrame:'):
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

                row_start_index = row - int(WINDOW_SIZE / 2)
                row_end_index = row_start_index + WINDOW_SIZE

                col_start_index = col - int(WINDOW_SIZE / 2)
                col_end_index = col_start_index + WINDOW_SIZE

                if row_start_index < 0 or col_start_index < 0 or row_end_index > STUDY_AREA_SHAPE[0] or col_end_index > STUDY_AREA_SHAPE[1]:
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
                        "start_row": row_start_index,
                        "end_row": row_end_index,
                        "start_col": col_start_index,
                        "end_col": col_end_index,
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


def parse_func(series, **kwarg):
    data_dict = defaultdict(list)

    with_t_1 = kwarg.get('with_t_1', True)

    start_row = series["start_row"]
    end_row = series["end_row"]
    start_col = series["start_col"]
    end_col = series["end_col"]

    past_h5_reader = H5BasicReader(series["t-10_H8"])
    for b_name in H8_BANDS:
        tmp_data = past_h5_reader.read(b_name)[start_row:end_row, start_col:end_col]
        data_dict['t-10_H8'].append([normalization(tmp_data, "H8")])

    if with_t_1:
        past_pre = GeoTIFFReader(series["t-10_pre"]).read()[start_row:end_row, start_col:end_col]
        data_dict['t-10_pre'].append([normalization(past_pre, "pre")])

    day_time = np.zeros(tmp_data.shape)
    day_time[:, :] = cosine_time_radians(series["t"], "day_time")
    data_dict['day_time'].append([day_time])

    year_time = np.zeros(tmp_data.shape)
    year_time[:, :] = cosine_time_radians(series["t"], "year_time")
    data_dict['year_time'].append([year_time])

    if with_t_1:
        data_dict['t-10_loc'].append(
            [
                normalization(GeoTIFFReader(series["t-10_loc"]).read()[start_row:end_row, start_col:end_col], "loc")
            ]
        )

    cur_h5_reader = H5BasicReader(series["t_H8"])
    for b_name in H8_BANDS:
        tmp_data = cur_h5_reader.read(b_name)[start_row:end_row, start_col:end_col]
        data_dict['H8'].append([normalization(tmp_data, "H8")])

    data_dict['t_dem'].append(
        [
            normalization(GeoTIFFReader(series["t_dem"]).read()[start_row:end_row, start_col:end_col], "dem")
        ]
    )

    pre = GeoTIFFReader(series["t_pre"]).read()[start_row:end_row, start_col:end_col]
    data_dict['pre'].append([normalization(pre, "pre")])
    return data_dict

def train_test_split(files_df_path, top_data_dir, train_size=(0.8, 0.1, 0.1), train=True, p_step=(1, 1)):
    if os.path.exists(files_df_path):
        files_df = pd.read_csv(files_df_path)
    else:
        files_df = get_items(top_data_dir)
        files_df.to_csv(files_df_path, index=False)

    unique_time = files_df["t"].unique()
    lens = len(unique_time)
    train_lens = int(np.ceil(lens * train_size[0]))
    valid_lens = int(np.ceil((lens * train_size[1])))

    train_time = unique_time[:train_lens]
    valid_time = unique_time[train_lens:train_lens + valid_lens]
    test_time = unique_time[train_lens + valid_lens:]

    train_files_df = files_df[files_df["t"].isin(train_time)]
    valid_files_df = files_df[files_df["t"].isin(valid_time)]
    if train:
        return train_files_df, valid_files_df
    else:
        test_files_df = files_df[files_df["t"].isin(test_time)]
        test_files_df = convert_test_df_for_demo(test_files_df, p_step)
        return test_files_df


def convert_test_df_for_demo(test_files_df, p_step):
    unique_time = test_files_df["t"].unique()
    tdf = pd.DataFrame(
        columns=[
            "t", "t-10_H8", "t-10_pre", "t-10_loc", "t_H8", "t_dem",
            "t_loc", "t_pre", "start_row", "end_row", "start_col", "end_col"
        ]
    )

    r = np.arange(0, STUDY_AREA_SHAPE[0], p_step[0])
    r = np.clip(r, 0, STUDY_AREA_SHAPE[0] - p_step[0])
    c = np.arange(0, STUDY_AREA_SHAPE[1], p_step[1])
    c = np.clip(c, 0, STUDY_AREA_SHAPE[1] - p_step[1])
    r, c = np.meshgrid(r, c)
    r_count = r.ravel().shape[0]
    for time in tqdm.tqdm(unique_time, desc='prepare test df:'):
        new_tdf = test_files_df[test_files_df['t'] == time].iloc[0:1, :]
        new_tdf = pd.concat([new_tdf] * r_count, ignore_index=True)
        new_tdf.loc[:, 'start_row'] = r.ravel()
        new_tdf.loc[:, 'end_row'] = r.ravel() + WINDOW_SIZE
        new_tdf.loc[:, 'start_col'] = c.ravel()
        new_tdf.loc[:, 'end_col'] = c.ravel() + WINDOW_SIZE
        tdf = pd.concat([tdf, new_tdf], ignore_index=True)
    return tdf


def data_generator(files_df_path, top_data_dir, batch_size,
                   train_size=(0.8, 0.1, 0.1), train=True, with_t_1=True, p_step=(1, 1)):
    if train:
        train_df, valid_df = train_test_split(files_df_path, top_data_dir, train_size, train=train)
        train_gen = DataGenerator(train_df, parse_func, batch_size, with_t_1=with_t_1)
        valid_gen = DataGenerator(valid_df, parse_func, batch_size, with_t_1=with_t_1)
        return train_gen, valid_gen
    else:
        test_df = train_test_split(files_df_path, top_data_dir, train_size, train=train)
        test_gen = DataGenerator(test_df, parse_func, batch_size, with_t_1=with_t_1, p_step=p_step)
        return test_gen
