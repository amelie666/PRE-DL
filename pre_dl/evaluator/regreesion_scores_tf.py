import numpy as np
from scipy import stats
import tensorflow as tf


class CalculateRegression(object):

    def __init__(self, y_obs, y_est):
        """
        :param y_obs: observation value
        :param y_est: estimation value
        """
        self.y_obs = y_obs
        self.y_est = y_est

    def calculate_me(self):
        """

        :return: 平均误差Mean Error (ME)
        """
        mean = np.mean(self.y_obs.flatten() - self.y_est.flatten())
        return round(mean, 4)

    def calculate_mae(self):
        """

        :return: 平均绝对误差Mean Absolute Error (MAE)
        """
        mae_tem = tf.losses.mean_absolute_error(self.y_obs, self.y_est)
        mae = mae_tem.numpy()
        return round(mae, 4)

    def calculate_rmse(self):
        """

        :return: 均方根误差Root Mean Squared Error (RMSE)
        """
        mse = np.sqrt(tf.losses.mean_squared_error(self.y_obs, self.y_est))
        return round(mse, 4)

    def calculate_rv(self):
        """

        :return: 方差缩小Reduction of Variance (RV)
        """
        # RV = 1-MSE?????
        y_obs_var = np.var(self.y_obs)
        rv = 1 - np.sqrt(tf.losses.mean_squared_error(self.y_obs, self.y_est)) / y_obs_var
        return round(rv,4)

    def calculate_pcorr(self):
        """

        :return: 皮尔逊相关系数Pearson Correlation Coefficient (PCORR)
        """
        # 数组size降为一维，只适用于array数组，不支持列表

        y1 = self.y_obs.flatten()
        y2 = self.y_est.flatten()
        # 两者的相关系数
        r, p = stats.pearsonr(y1, y2)
        return r

    def calculate_scorr(self):
        """
        :return: Spearman等级相关性the Spearman Rank Correlation (SCORR)
        """
        y1 = self.y_obs.flatten()
        y2 = self.y_est.flatten()
        r, p = stats.spearmanr(y1, y2, )
        return r
