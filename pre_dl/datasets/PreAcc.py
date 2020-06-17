import os

import requests
import pandas as pd


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

    def save(self, fpath: str):
        """

        :param fpath: csv file path
        """
        df = pd.DataFrame(data=self.content['DS'])
        df.to_csv(fpath, index=False)
