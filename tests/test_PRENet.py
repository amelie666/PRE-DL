import os
import unittest

from pre_dl.model.PRENet import PRENet

from tensorflow.keras.utils import plot_model

class TestPRENet(unittest.TestCase):
    
    def test_PRENetPlot(self):
        cnn = PRENet().nn(input_shape=(512, 512, 3), valid_rain=(0, 100))
        plot_model(cnn, 'tmp/DNNs.png',  show_shapes=True)
