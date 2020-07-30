import os
import argparse

import numpy as np

from pyproj import Proj, transform
import rasterio as rio


def main():
    parser = argparse.ArgumentParser(description="Project and Clip DEM tif file.")
    parser.add_argument('full_crs',
                        type=str,
                        help='raw dem crs')
    parser.add_argument('full_extent',
                        type=float,
                        nargs='+',
                        help='raw dem extent: lat_min, lat_max, lon_min, lon_max')
    parser.add_argument('region_crs',
                        type=str,
                        help='region dem crs')
    parser.add_argument('resolution',
                        type=int,
                        default=2000,
                        help='region dem resolution')
    parser.add_argument('--workspace',
                        '-w',
                        type=str,
                        default='.',
                        help='process directory')
    args = parser.parse_args()
    # p_dst_proj4 = '+proj=longlat'
    p_dst_proj4 = args.full_crs

    # lat_min, lat_max, lon_min, lon_max = 36, 42.075, 111.96, 120
    lat_min, lat_max, lon_min, lon_max = args.full_extent

    # p_src_proj4 = '+proj=aea +lat_1=29.5 +lat_2=45.5 +lon_0=116'
    p_src_proj4 = args.region_crs
    # p_src_res = 2000
    p_src_res = args.resolution

    # workspace = r'D:\WorkSpace\Process\DEM'
    workspace = args.workspace

    p_src = Proj(p_src_proj4)
    p_dst = Proj(p_dst_proj4)

    lat_b, lon_b = np.meshgrid([lat_min, lat_max], [lon_min, lon_max])
    lon_b = np.asarray(lon_b.ravel().tolist() + [(lon_min + lon_max) / 2])
    lat_b = np.asarray(lat_b.ravel().tolist() + [lat_min])

    v_b, u_b = p_src(lon_b, lat_b)

    v = np.arange(v_b.min(), v_b.max(), p_src_res)
    u = np.arange(u_b.max(), u_b.min(), -p_src_res)

    v_src, u_src = np.meshgrid(v, u)

    lon_src, lat_src = transform(p_src, p_dst, v_src, u_src)

    print("longitude", lon_src.min(), lon_src.max())
    print("latitude", lat_src.min(), lat_src.max())
    print("output array shape:", lon_src.shape)

    with rio.open(
        os.path.join(workspace, "DEM.tif")
    ) as f:
        _, lon_res, _, _, _, lat_res = f.get_transform()
        array = f.read(1)

    c_i = np.round((lon_src - f.bounds.left) / lon_res).astype(np.int)
    r_i = np.round((lat_src - f.bounds.top) / lat_res).astype(np.int)

    dem_dst = np.full(v_src.shape, -999)
    dem_dst[:, :] = array[r_i, c_i]
    with rio.open(
        os.path.join(workspace, 'H8_DEM.tif'),
        'w',
        driver='GTiff',
        height=dem_dst.shape[0],
        width=dem_dst.shape[1],
        count=1,
        dtype=dem_dst.dtype.name,
        crs=p_src_proj4,
        transform=rio.Affine.from_gdal(v_src[0, 0], p_src_res, 0.0, u_src[0, 0], 0.0, -p_src_res),
        nodata=(-999)
    ) as dst:
        dst.write(dem_dst, 1)


if __name__ == '__main__':
    main()
