from pre_dl.datasets.PreAcc import PreAcc

import datetime
import os
import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser(description="Project SURF_WEA_CHN_PRE_MIN_ACCU csv file.")
    parser.add_argument('start_time',
                        type=str,
                        help='start time in \'%Y%m%d_%H%M%S\' format ')
    parser.add_argument('count',
                        type=int,
                        help='count')
    parser.add_argument('full_crs',
                        type=str,
                        help='raw prep crs')
    parser.add_argument('full_extent',
                        type=float,
                        nargs='+',
                        help='raw prep extent: lat_min, lat_max, lon_min, lon_max')
    parser.add_argument('region_crs',
                        type=str,
                        help='region prep crs')
    parser.add_argument('resolution',
                        type=int,
                        default=2000,
                        help='region prep resolution')
    parser.add_argument('--workspace',
                        '-w',
                        type=str,
                        default='.',
                        help='process directory')
    parser.add_argument('--prepout',
                        '-po',
                        type=str,
                        default='.',
                        help='prep output directory')
    parser.add_argument('--locout',
                        '-lo',
                        type=str,
                        default='.',
                        help='location output directory')
    parser.add_argument('--rgpout',
                        '-ro',
                        type=str,
                        default='.',
                        help='rain gauge output directory')
    args = parser.parse_args()
    start_dt = datetime.datetime.strptime(args.start_time, '%Y%m%d_%H%M%S')
    interval = datetime.timedelta(minutes=10)
    total = args.total
    cursor_dt = start_dt
    os.makedirs(args.prepout, exist_ok=True)
    os.makedirs(args.locout, exist_ok=True)
    os.makedirs(args.rgout, exist_ok=True)
    for _ in tqdm.trange(total, desc='Processing PRE'):
        pa = PreAcc()
        raw_stat_csv = os.path.join(args.workspace, cursor_dt.strftime('%Y%m%d%H%M%S.csv'))
        prep_tif = os.path.join(args.prepout, cursor_dt.strftime('%Y%m%d%H%M_Pre.tif'))
        loc_tif = os.path.join(args.locout, cursor_dt.strftime('%Y%m%d%H%M_Location.tif'))
        stat_csv = os.path.join(args.rgout, cursor_dt.strftime('%Y%m%d%H%M.csv'))
        pa.load_csv(raw_stat_csv)
        pa.project_to_ds(prep_tif, loc_tif, stat_csv, args.region_crs, args.full_extent, args.resolution)
        cursor_dt += interval


if __name__ == '__main__':
    main()
