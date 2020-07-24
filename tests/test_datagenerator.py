import os
import unittest

from pre_dl.datasets.data_generator import data_generator


class TestDatasets(unittest.TestCase):
    
    def test_PRENetPlot(self):
        nn_data_dir = r"/hxqtmp/DPLearning/hm/data/PRE"
        train_gen, valid_gen, test_gen = data_generator(nn_data_dir, 1)
        x,  y = train_gen[0]
        for key in x.keys():
            print(key, x[key].shape)
        for key in y.keys():
            print(key, y[key].shape)
        