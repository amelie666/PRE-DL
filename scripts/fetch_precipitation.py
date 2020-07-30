from pre_dl.datasets.PreAcc import PreAcc

import datetime
import os
import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser(description="DownLoad SURF_WEA_CHN_PRE_MIN_ACCU from CIMISS.")
    parser.add_argument('start_time',
                        type=str,
                        help='start time in \'%Y%m%d_%H%M%S\' format ')
    parser.add_argument('count',
                        type=int,
                        help='count')
    parser.add_argument('--out',
                        '-o',
                        type=str,
                        default='.',
                        help='output directory')
    args = parser.parse_args()
    start_dt = datetime.datetime.strptime(args.start_time, '%Y%m%d_%H%M%S')
    interval = datetime.timedelta(minutes=10)
    total = args.total
    cursor_dt = start_dt
    os.makedirs(args.out, exist_ok=True)
    for _ in tqdm.trange(total, desc='Downloading PRE'):
        pa = PreAcc()
        pa.fetch(cursor_dt.strftime('%Y%m%d%H%M%S'))
        cursor_file_path = os.path.join(args.out, cursor_dt.strftime('%Y%m%d%H%M%S.csv'))
        pa.to_csv(cursor_file_path)
        cursor_dt += interval


if __name__ == '__main__':
    main()
