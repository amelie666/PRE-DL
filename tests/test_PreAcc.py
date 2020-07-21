import os
import unittest

from pre_dl.datasets.PreAcc import PreAcc, PMMMin


class TestDataSet(unittest.TestCase):
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

    def test_PMMMinSave(self):
        cursor_datetime = '20190101000000'
        cursor_fpath = 'tmp/20190101000000_PMM.csv'
        pm = PMMMin()
        pm.fetch(cursor_datetime)
        pm.save(cursor_fpath)
        self.assertTrue(os.path.exists(cursor_fpath))

    def test_PMMMinPlot(self):
        cursor_datetime = '20200713120000'
        pm = PMMMin()
        pm.fetch(cursor_datetime)
        pm.plot()


if __name__ == '__main__':
    unittest.main()
