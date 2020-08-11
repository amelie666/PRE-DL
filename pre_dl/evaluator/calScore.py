#coding:utf-8
import numpy as np
from scipy import stats
import tensorflow as tf
import csv
from math import sqrt


class CalScores(object):
	def __init__(self, y_true, y_pre):
		self.x = y_true
		self.y = y_pre

	def calRegScores(self):
		mean = np.mean(self.x.flatten() - self.y.flatten())
		ME = round(mean, 4)

		mae_tem = tf.losses.mean_absolute_error(self.x, self.y)
		mae = mae_tem.numpy()
		MAE = round(mae, 4)

		mse = tf.losses.mean_squared_error(self.x, self.y)
		rmse = np.sqrt(mse)
		RMSE = round(rmse, 4)

		x = np.var(self.x)
		rv = 1 - mse / x
		RV = round(rv, 4)

		x = self.x.flatten()
		y = self.y.flatten()
		pcorr, _ = stats.pearsonr(x, y)
		scorr, _ = stats.spearmanr(x, y)

		return ME, MAE, RMSE, RV, pcorr, scorr

	def calClassScores(self,threshold=20):
		x = self.x
		y = self.y
		x[x >= threshold] = 1
		x[x < threshold] = 0
		y[y >= threshold] = 1
		y[y < threshold] = 0

		tn, fp, fn, tp = tf.math.confusion_matrix(x, y, num_classes=2).numpy().ravel()


		# pod
		if tp + fn == 0.0:
			pod = 0.0
		else:
			pod = tp / (tp + fn)

		# far
		if fp + tp == 0.0:
			far = 0.0
		else:
			far = fp / (fp + tp)

		# pofd
		if fp + tn == 0:
			pofd = 0.0
		else:
			pofd = fp / (fp + tn)

		# acc
		acc = (tp + tn) / (tp + tn + fp + fn)

		# csi
		csi = tp / (tp + fp + fn)

		# gss
		tpr = (tp + fp) * (tn + fn) / (tp + fp + tn + fn)
		gss = (tp - tpr) / (tp + fp + fn - tpr)

		# hss
		hss = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (tp + fp + tn + fn)

		# hkd
		if tp + fn == 0.0 or fp + tn == 0.0:
			hkd = 0.0
		else:
			hkd = (tp / (tp + fn)) + (fp / (fp + tn))
		# f1
		if tp + tn == 0.0 or tp + fn == 0.0:
			f1 = 0.0
		else:
			P = tp / (tp + tn)
			R = tp / (tp + fn)
			f1 = 2 * P * R / (P + R + 0.000001)
		# iou
		iou = tp / (tp + fn + fp)

		return pod, far, pofd, acc, csi, gss, hss, hkd, f1, iou


	def saveFiles(self, filename,**kwargs):#**kwargs把多个关键字参数打包成字典
		with open(filename,'w',newline='') as f:
			csv_writer = csv.writer(f)
			for l in [kwargs]:
				for k, v in l.items():
					# print(k, v)
					csv_writer.writerow([k, v])




