import os
import unittest

from pre_dl.datasets.PmmMin import PmmMin, PmmMinTimeRange


class TestPmmMin(unittest.TestCase):

    def test_PmmMinSave(self):
        cursor_datetime = '20200722090000'
        cursor_fpath = 'tmp/%s_PMM.csv' % cursor_datetime
        pm = PmmMin()
        pm.fetch(cursor_datetime)
        pm.save(cursor_fpath)
        self.assertTrue(os.path.exists(cursor_fpath))

    def test_PmmMinTimeRangeSave(self):
        range_datetime = '[20200722090000,20200722091000]'
        cursor_fpath = 'tmp/%s_PMM.csv' % range_datetime
        pm = PmmMinTimeRange()
        pm.fetch(range_datetime)
        pm.save(cursor_fpath)
        self.assertTrue(os.path.exists(cursor_fpath))

    def test_PmmMinPlot(self):
        cursor_datetime = '20200722090000'
        pm = PmmMin()
        pm.fetch(cursor_datetime)
        pm.plot()


if __name__ == '__main__':
    unittest.main()
