# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:50:03 2020

@author: Marek
"""


from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from linear_regression import linear_regression as lr

x, y = make_regression(n_samples = 1500, n_features=20, n_informative=6)
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.1, random_state=5, shuffle=True)
lin = lr.LinearRegression(x_tr, y_tr, max_iter=1000, reg_lambda=0)
lin.fit()
loss = lin.costs

sk = Lasso(max_iter=1000)
sk.fit(x_tr, y_tr)

mae1 = mean_absolute_error(y_te, lin.predict(x_te))
mae2 = mean_absolute_error(y_te, sk.predict(x_te))

plt.plot(range(len(loss)), loss)
    