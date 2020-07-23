import os
import unittest

from pre_dl.model.PRENet import PRENet

from tensorflow.keras.utils import plot_model

import warnings

warnings.filterwarnings("ignore")

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 

class TestPRENet(unittest.TestCase):
    
    def test_PRENetPlot(self):
        cnn = PRENet().compile(main_input_shape=(344, 360, 12), gt_ppre_input_shape=(344, 360, 1),gt_pre_input_shape=(344, 360, 1), valid_rain=(0, 100), lr=1e-3, lr_decay=1e-4)
        # plot_model(cnn, 'tmp/DNNs.png',  show_shapes=True)
