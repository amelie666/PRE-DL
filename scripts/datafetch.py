from pre_dl.datasets.PreAcc import PreAcc

import datetime
import os
import tqdm
import argparse


def main(start_time, count, out):
    interval = datetime.timedelta(minutes=10)
    cursor_dt = start_time
    for i in tqdm.trange(count, desc='Downloading PRE'):
        try:
            pa = PreAcc()
            pa.fetch(cursor_dt.strftime('%Y%m%d%H%M%S'))
            cursor_fpath = os.path.join(out, cursor_dt.strftime('%Y%m%d%H%M%S.csv'))
            pa.save(cursor_fpath)
            cursor_dt += interval
        except Exception as e:
            print(e, cursor_dt, i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DownLoad SURF_WEA_CHN_PRE_MIN_ACCU from CIMISS.")
    parser.add_argument('start_time',
                        type=str,
                        help='start time')
    parser.add_argument('count',
                        type=int,
                        help='count')
    parser.add_argument('--out',
                        '-o',
                        type=str,
                        default='.',
                        help='output directory')
    args = parser.parse_args()
    main(args.start_time, args.count, args.out)
