import os
import unittest

import pandas as pd
import dask as da
import numpy as np
from pre_dl.datasets.data_generator import DataGenerator,parse_func


class TestDatasets(unittest.TestCase):
    
    def test_data_generator_train_val(self):
        bs = 1
        train_df_path= './data/round4/9x9_patchs_train.csv'
        val_df_path= './data/round4/9x9_patchs_val.csv'
        top_data_dir = "/hxqtmp/DPLearning/hm/data/PRE"
        train_df = pd.read_csv(train_df_path)
        val_df = pd.read_csv(val_df_path)
        train_gen = DataGenerator(train_df, parse_func, bs)
        valid_gen = DataGenerator(val_df, parse_func, bs)
        x = train_gen[0]
        print('train x:')
        for x_item in x:
           for key in x_item.keys():
                print(key, len(x_item[key]), np.asarray((x_item[key])).shape)
            
    def test_data_generator_test(self):
        bs = 1
        test_df_path= './data/round4/9x9_patchs_test.csv'
        top_data_dir = "/hxqtmp/DPLearning/hm/data/PRE"
        test_df_iterator = pd.read_csv(test_df_path, chunksize=118272)
        test_df = next(test_df_iterator)
        test_gen = DataGenerator(test_df, parse_func, bs)
        x = test_gen[0]
        print('train x:')
        for x_item in x:
           for key in x_item.keys():
                print(key, len(x_item[key]), np.asarray((x_item[key])).shape)
            
        time = test_gen.get_time(0)[0]
        print('time', time)
        rows = test_gen.get_row(0)[0]
        print('rows', rows)
        cols = test_gen.get_col(0)[0]
        print('cols', cols)

        