import os

import joblib
from sklearn.linear_model import LinearRegression


class QPELR(object):

    def __init__(self):
        self._clf = LinearRegression()

    def fit(self, x, y):
        self._clf.fit(x, y)

    def save_model(self, filepath):
        dir_path = os.path.dirname(filepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        joblib.dump(self._clf, filepath)

    def show_equation(self, filepath=None):
        coef = self._clf.coef_
        intercpet = self._clf.intercept_
        coef_lst = [str(i) for i in coef]
        var_lst = ["x_{}".format(i+1) for i in range(len(coef))]
        equation_lst = [" * ".join(i) for i in list(zip(coef_lst, var_lst))]

        equation = "y = " + " + ".join(equation_lst) + " + {}".format(intercpet)

        print("Equation: ", equation)

        if filepath is not None:

            dir_path = os.path.dirname(filepath)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            with open(filepath, "w") as f:
                f.write(equation)

    def predict(self, x, model_path):
        if os.path.exists(model_path):
            clf = joblib.load(model_path)
        else:
            clf = self._clf

        return clf.predict(x)
