import os
import unittest

from pre_dl.datasets.data_generator import data_generator


class TestDatasets(unittest.TestCase):
    
    def test_data_generator_train_val(self):
        bs = 1
        files_df_path= './data/round4/9x9_patchs.csv'
        top_data_dir = "/hxqtmp/DPLearning/hm/data/PRE"
        train_gen, valid_gen = data_generator(files_df_path, top_data_dir, bs)
        x,  y = train_gen[0]
        print('train x:')
        for key in x.keys():
            print(key, x[key].shape)
        print('train y:')
        for key in y.keys():
            print(key, y[key].shape)
            
        x,  y = valid_gen[0]
        print('train x:')
        for key in x.keys():
            print(key, x[key].shape)
        print('train y:')
        for key in y.keys():
            print(key, y[key].shape)
            
    def test_data_generator_test(self):
        bs = 1
        files_df_path= './data/round4/9x9_patchs.csv'
        top_data_dir = "/hxqtmp/DPLearning/hm/data/PRE"
        test_gen = data_generator(files_df_path, top_data_dir, bs, train=False)
        x,  y = test_gen[0]
        print('test x:')
        for key in x.keys():
            print(key, x[key].shape)
        print('test y:')
        for key in y.keys():
            print(key, y[key].shape)
            
        time = test_gen.get_time(0)[0]
        print('time', time)
        rows = test_gen.get_row(0)[0]
        print('rows', rows)
        cols = test_gen.get_col(0)[0]
        print('cols', cols)
        
    def test_data_generator_test(self):
        bs = 1
        files_df_path= '/hxqtmp/DPLearning/bupj/PRE_DL/debug/tmp/NCHN_mon_5.csv'
        top_data_dir = "/hxqtmp/DPLearning/hm/data/PRE"
        test_gen = data_generator(files_df_path, top_data_dir, bs, train=False, with_t_1=False)
        x,  y = test_gen[0]
        print('test x:')
        for key in x.keys():
            print(key, x[key].shape)
        print('test y:')
        for key in y.keys():
            print(key, y[key].shape)
            
        time = test_gen.get_time(0)[0]
        print('time', time)
        rows = test_gen.get_row(0)[0]
        print('rows', rows)
        cols = test_gen.get_col(0)[0]
        print('cols', cols)
        