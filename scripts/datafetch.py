from pre_dl.datasets.PreAcc import PreAcc

import datetime
import os
import tqdm


def main():
    start_dt = datetime.datetime(2019, 1, 1, 0, 0, 0)
    interval = datetime.timedelta(minutes=10)
    total = int(365 * 24 * 60 / 10)
    cursor_dt = start_dt
    for i in tqdm.trange(368, total, desc='Downloading PRE'):
        pa = PreAcc()
        pa.fetch(cursor_dt.strftime('%Y%m%d%H%M%S'))
        cursor_fpath = os.path.join(r'F:\research\PRE\PRE_ACC_CSV', cursor_dt.strftime('%Y%m%d%H%M%S.csv'))
        pa.save(cursor_fpath)
        cursor_dt += interval


if __name__ == '__main__':
    main()
