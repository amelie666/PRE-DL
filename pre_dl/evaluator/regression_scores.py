from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from scipy import stats


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
        return np.mean(self.y_obs.flatten() - self.y_est.flatten())

    def calculate_mae(self):
        """

        :return: 平均绝对误差Mean Absolute Error (MAE)
        """
        mae_tem = mean_absolute_error(self.y_obs, self.y_est)
        mae = round(mae_tem, 2)
        return mae

    def calculate_rmse(self):
        """

        :return: 均方根误差Root Mean Squared Error (RMSE)
        """
        return np.sqrt(mean_squared_error(self.y_obs, self.y_est))

    def calculate_rv(self):
        """

        :return: 方差缩小Reduction of Variance (RV)
        """
        # RV = 1-MSE?????
        y_obs_var = np.var(self.y_obs)
        return 1 - np.sqrt(mean_squared_error(self.y_obs, self.y_est)) / y_obs_var

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
        r, p = stats.spearmanr(y1, y2)
        return r
