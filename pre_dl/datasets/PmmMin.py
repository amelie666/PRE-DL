from .CiMissDownloader import CiMissDownloader

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class PmmMin(CiMissDownloader):
    content = None

    def __init__(self):
        super(PmmMin, self).__init__()
        data_params = {'interfaceId': 'getCawnEleByTime',
                       'dataCode': 'CAWN_CHN_PMM_MIN',
                       'elements': 'Station_Id_C,Lat,Lon,Year,Mon,Day,Hour,Min,PM2p5_Densty',
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
        valid1 = df['PM2p5_Densty'] != '999999'
        valid2 = df['PM2p5_Densty'] != '-999.9'
        df_x = df[valid1 & valid2]
        # lat 70-0  long 80-140
        grid_x, grid_y = np.mgrid[lat_max:lat_min:2800j, lon_min:lon_max:2400j]
        points = df_x[['Lat', 'Lon']].values.astype(np.float32)
        values = df_x['PM2p5_Densty'].values.astype(np.float32)
        grid_z1 = interpolate.griddata(points, values, (grid_x, grid_y), method=method)
        return points, grid_z1
