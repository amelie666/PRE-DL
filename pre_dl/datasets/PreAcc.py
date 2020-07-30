from .CiMissDownloader import CiMissDownloader

import requests
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt


class PreAcc(CiMissDownloader):
    content = None
    df: pd.DataFrame

    def __init__(self):
        super(PreAcc, self).__init__()
        data_params = {'interfaceId': 'getSurfEleByTime',
                       'dataCode': 'SURF_WEA_CHN_PRE_MIN_ACCU',
                       'elements': 'Station_Id_C,Lat,Lon,Year,Mon,Day,Hour,Min,V13392_010,Q_V13392_010',
                       'orderby': 'Station_ID_C:ASC',
                       'dataFormat': 'json'
                       }
        self.params.update(data_params)

    def fetch(self, datetime: str):
        """

        :param datetime:  example: 20190101000000
        """
        self.params.update({'times': datetime})
        r = requests.get(self.cimiss_url, params=self.params)
        self.content = r.json()
        self.df = pd.DataFrame(data=self.content['DS'])

    def load_csv(self, csv_fpath: str):
        self.df = pd.read_csv(csv_fpath)

    def to_csv(self, csv_fpath: str):
        self.df.to_csv(csv_fpath, index=False)

    def project(self, dst_crs: str, src_extent: list, dst_res=2000, aggregate=True):
        from pyproj import Proj, transform
        p_dst = Proj('+proj=longlat')
        p_src = Proj(dst_crs)
        lat_min, lat_max, long_min, long_max = src_extent
        lat_b, long_b = np.meshgrid([lat_min, lat_max], [long_min, long_max])
        long_b = np.asarray(long_b.ravel().tolist() + [(long_min + long_max) / 2])
        lat_b = np.asarray(lat_b.ravel().tolist() + [lat_min])
        v_b, u_b = p_src(long_b, lat_b)
        v = np.arange(v_b.min(), v_b.max(), dst_res)
        u = np.arange(u_b.max(), u_b.min(), -dst_res)

        valid = self.df['Q_V13392_010'] == 0

        df = self.df[valid]
        df['v'], df['u'] = transform(p_dst, p_src, df['Lon'].values, df['Lat'].values)
        c1 = df['u'] >= u_b.min()
        c2 = df['u'] < u_b.max()
        c3 = df['v'] >= v_b.min()
        c4 = df['v'] < v_b.max()
        c = np.stack([c1, c2, c3, c4])
        c = np.prod(c, 0)
        df = df[c.astype(np.bool)]

        df['v_i'] = np.floor((df['v'] - v_b.min()) / dst_res).astype(np.int)
        df['u_i'] = np.floor((u_b.max() - df['u']) / dst_res).astype(np.int)
        if aggregate:
            df['u_i_s'] = df['u_i']
            df['v_i_s'] = df['v_i']
            df = df.groupby(['u_i_s', 'v_i_s']).mean()
        return df, (u, v, dst_res)

    def project_to_ds(self,
                      prep_tif: str,
                      loc_tif: str,
                      stat_csv: str,
                      dst_crs: str,
                      src_extent: list,
                      dst_res=2000,
                      aggregate=True):
        import rasterio as rio
        df, (u, v, dst_res) = self.project(dst_crs, src_extent, dst_res, aggregate)
        prep = np.zeros((u.shape[0], v.shape[0]), dtype=np.float32)
        prep[df[['u_i']].values, df[['v_i']].values] = df[['V13392_010']].values
        with rio.open(
            prep_tif,
            'w',
            driver='GTiff',
            height=prep.shape[0],
            width=prep.shape[1],
            count=1,
            dtype=prep.dtype.name,
            crs=dst_crs,
            transform=rio.Affine.from_gdal(v[0], dst_res, 0.0, u[0], 0.0, -dst_res),
        ) as dst:
            dst.write(prep, 1)
        loc = np.zeros((u.shape[0], v.shape[0]), dtype=np.uint8)
        loc[df[['u_i']].values, df[['v_i']].values] = 1
        with rio.open(
            loc_tif,
            'w',
            driver='GTiff',
            height=loc.shape[0],
            width=loc.shape[1],
            count=1,
            dtype=loc.dtype.name,
            crs=dst_crs,
            transform=rio.Affine.from_gdal(v[0], dst_res, 0.0, u[0], 0.0, -dst_res),
        ) as dst:
            dst.write(loc, 1)
        df.to_csv(stat_csv, index=False)

    def plot(self):
        points, grid_z1 = self._interp()
        plt.figure(figsize=(10, 10))
        plt.plot(points[:, 1], points[:, 0], 'ro', alpha=0.2)
        plt.imshow(grid_z1[::-1, ::], extent=(60, 140, 0, 70), origin='lower', cmap='jet')
        plt.show()

    def _interp(self, lat_min=0, lat_max=70, lon_min=60, lon_max=140, method='linear'):
        df = pd.DataFrame(data=self.content['DS'])
        valid = df['Q_V13392_010'] == '0'
        df_x = df[valid]
        # lat 70-0  long 80-140
        grid_x, grid_y = np.mgrid[lat_max:lat_min:2800j, lon_min:lon_max:2400j]
        points = df_x[['Lat', 'Lon']].values.astype(np.float32)
        values = df_x['V13392_010'].values.astype(np.float32)
        grid_z1 = interpolate.griddata(points, values, (grid_x, grid_y), method=method)
        return points, grid_z1
