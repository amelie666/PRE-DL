import os

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class CiMissDownloader(object):

    def __init__(self):
        self.cimiss_url = 'http://10.20.76.55/cimiss-web/api'
        self.params = {'userId': 'NSMC_KZS_ZHANGXZ',
                       'pwd': 'zhangxz123'}


class PreAcc(CiMissDownloader):
    content = None

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

    def save(self, fpath: str, format='csv'):
        """

        :param fpath: csv file path
        :param format: csv/jpg/png/hdf/nc
        """
        df = self._parse()
        if format == 'csv':
            df.to_csv(fpath, index=False)
        elif format == 'jpg':
            import PIL
            pass
        elif format == 'png':
            import PIL
            pass
        elif format == 'hdf':
            import h5py
            pass
        elif format == 'nc':
            import xarray as xr
            pass

    def plot(self):
        points, grid_z1 = self._interp()
        plt.figure(figsize=(10, 10))
        plt.plot(points[:, 1], points[:, 0], 'ro', alpha=0.2)
        plt.imshow(grid_z1[::-1, ::], extent=(60, 140, 0, 70), origin='lower', cmap='jet')
        plt.show()

    def _parse(self):
        df = pd.DataFrame(data=self.content['DS'])
        return df

    def _interp(self, lat_min=0, lat_max=70, lon_min=60, lon_max=140, method='linear'):
        df = self._parse()
        valid = df['Q_V13392_010'] == 0
        df_x = df[valid]
        # lat 70-0  long 80-140
        grid_x, grid_y = np.mgrid[lat_max:lat_min:2800j, lon_min:lon_max:2400j]
        points = df_x[['Lat', 'Lon']].values
        values = df_x['V13392_010'].values
        grid_z1 = interpolate.griddata(points, values, (grid_x, grid_y), method=method)
        return points, grid_z1
