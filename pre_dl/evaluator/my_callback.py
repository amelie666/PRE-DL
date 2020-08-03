import numpy as np
import tensorflow as tf
from tensorflow import keras
from regression_scores import CalculateRegression
from classification_scores import CalculateClassification


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def on_train_begin(self, logs={}):
        # 训练开始
        # print('开始？？')
        return

    def on_train_end(self, logs={}):
        # 训练结束 需要得到模型评估的图形
        # print('结束？？？')
        return

    def on_epoch_begin(self, epoch, logs={}):
        # 每一批次的开始
        # print('epoch 开始？')
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_test)
        val_mean = CalculateRegression(y_pred_val, self.y_test).calculate_me()
        val_rmse = CalculateRegression(y_pred_val, self.y_test).calculate_rmse()
        val_mae = CalculateRegression(y_pred_val, self.y_test).calculate_mae()
        val_rv = CalculateRegression(y_pred_val, self.y_test).calculate_rv()
        val_pcorr = CalculateRegression(y_pred_val, self.y_test).calculate_pcorr()
        val_scorr = CalculateRegression(y_pred_val, self.y_test).calculate_scorr()
        print(
            'On {} epoch end: {}val_mean:{:.4f} - val_rmse:{:.4f} - val_mae:{:.4f} - val_rc:{:.4f} - val_pcorr:{:.4f} - val_scorr:{:.4f} {}'
                .format(epoch, '{', val_mean, val_rmse, val_mae, val_rv, val_pcorr, val_scorr, '}'))

        # 降水量进行二值分类
        # 降水量的阈值
        y_pred_class = (y_pred_val >= 0).astype(np.int_)
        y_class = (self.y_test >= 0).astype(np.int_)
        tn, fp, fn, tp = tf.math.confusion_matrix(y_class, y_pred_class, 2).numpy().ravel()
        val_pod = CalculateClassification(tn, fp, fn, tp).calculate_pod()
        val_far = CalculateClassification(tn, fp, fn, tp).calculate_far()
        val_pofd = CalculateClassification(tn, fp, fn, tp).calculate_pofd()
        val_acc = CalculateClassification(tn, fp, fn, tp).calculate_acc()
        val_csi = CalculateClassification(tn, fp, fn, tp).calculate_csi()
        val_gss = CalculateClassification(tn, fp, fn, tp).calculate_gss()
        val_hss = CalculateClassification(tn, fp, fn, tp).calculate_hss()
        val_hkd = CalculateClassification(tn, fp, fn, tp).calculate_hkd()
        val_f1 = CalculateClassification(tn, fp, fn, tp).calculate_f1()
        print(
            '                 {}val_pod:{:.4f} - val_far:{:.4f} - val_pofd:{:.4f} - val_acc:{:.4f}- val_csi:{:.4f}'
            '- val_gss:{:.4f} - val_hss:{:.4f} - val_hkd:{:.4f} - val_f1:{:.4f} {}'
                .format('{', val_pod, val_far, val_pofd, val_acc, val_csi, val_gss, val_hss, val_hkd, val_f1, '}'))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

