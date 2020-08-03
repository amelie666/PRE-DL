class CalculateClassification(object):

    def __init__(self, tn, fp, fn, tp):
        """
        :param tp: the number of true positives
        :param tn: the number of true negatives
        :param fp: the number of false positives
        :param fn: the number of false negatives
        """
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.tp = tp

    def calculate_pod(self):
        """
        :return: 检测率Probability Of Detection (POD)
        """
        if self.tp + self.fn == 0.0:
            return 0.0
        else:
            return self.tp / (self.tp + self.fn)

    def calculate_far(self):
        """
        :return: False Alarm Rate 误报率
        """
        if self.fp + self.tp == 0.0:
            return 0.0
        else:
            return self.fp / (self.fp + self.tp)

    def calculate_pofd(self):
        """
        :return: Probability of False Detection 误报概率
        """
        if self.fp + self.tn == 0:
            return 0.0
        else:
            return self.fp / (self.fp + self.tn)

    def calculate_acc(self):
        """
        :return: ACCuracy准确率
        """
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def calculate_csi(self):
        """
        :return:  成功关键指数Critical Success Index (CSI)
        """
        return self.tp / (self.tp + self.fp + self.fn)

    def calculate_gss(self):
        """
        :return:  吉尔伯特技能分数Gilbert Skill Score (GSS)
        """
        tpr = (self.tp + self.fp) * (self.tn + self.fn) / (self.tp + self.fp + self.tn + self.fn)
        return (self.tp - tpr) / (self.tp + self.fp + self.fn - tpr)

    def calculate_hss(self):
        """
        :return:  海德克技能分数Heidke Skill Score (HSS
        """
        return (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn) / (
                self.tp + self.fp + self.tn + self.fn)

    def calculate_hkd(self):
        """

        :return:  汉森-奎珀斯判别器Hanssen–Kuipers Discriminant (HKD)
        """
        if self.tp + self.fn == 0.0 or self.fp + self.tn == 0.0:
            return 0.0
        else:
            return (self.tp / (self.tp + self.fn)) + (self.fp / (self.fp + self.tn))

    def calculate_f1(self):
        """
        :return: F1分数F1 score (F1)
        """
        if self.tp + self.tn == 0.0 or self.tp + self.fn == 0.0:
            return 0.0
        else:
            P = self.tp / (self.tp + self.tn)
            R = self.tp / (self.tp + self.fn)
            return 2 * P * R / (P + R + 0.000001)

    def calculate_iou(self):
        iou = self.tp / (self.tp + self.fn + self.fp)
        return iou
