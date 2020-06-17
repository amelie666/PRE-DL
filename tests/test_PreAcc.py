import os
import unittest

from pre_dl.datasets.PreAcc import PreAcc


class TestDataSet(unittest.TestCase):
    def test_PreAcc(self):
        cursor_datetime = '20190101000000'
        cursor_fpath = 'tmp/20190101000000.csv'
        pa = PreAcc()
        pa.fetch(cursor_datetime)
        pa.save(cursor_fpath)
        self.assertTrue(os.path.exists(cursor_fpath))


if __name__ == '__main__':
    unittest.main()
