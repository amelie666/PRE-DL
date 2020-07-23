import os
import unittest

from pre_dl.datasets.PreAcc import PreAcc


class TestPreAcc(unittest.TestCase):

    def test_PreAccSave(self):
        cursor_datetime = '20190101000000'
        cursor_fpath = 'tmp/20190101000000_PRE.csv'
        pa = PreAcc()
        pa.fetch(cursor_datetime)
        pa.save(cursor_fpath)
        self.assertTrue(os.path.exists(cursor_fpath))

    def test_PreAccPlot(self):
        cursor_datetime = '20190101000000'
        pa = PreAcc()
        pa.fetch(cursor_datetime)
        pa.plot()


if __name__ == '__main__':
    unittest.main()
