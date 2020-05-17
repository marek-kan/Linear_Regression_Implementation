# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:50:03 2020

@author: Marek
"""


from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from linear_regression import linear_regression as lr
    
x, y = load_boston(return_X_y=True)
scaler = StandardScaler()

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.1, random_state=5, shuffle=True)
x_tr = scaler.fit_transform(x_tr)
x_te = scaler.transform(x_te)

lin = lr.LinearRegression(max_iter=5000, learning_rate=0.01, reg_lambda=5e-5, beta=0.6)
lin.fit(x_tr, y_tr)
loss = lin.costs

sk = Lasso(max_iter=5000, alpha=5e-5)
sk.fit(x_tr, y_tr)

mae1 = mean_absolute_error(y_te, lin.predict(x_te))
mae2 = mean_absolute_error(y_te, sk.predict(x_te))

plt.plot(range(len(loss)), loss)
plt.title('Training loss')
plt.show()
plt.close()

# Test "online learning"
new_examples = 5
online_mae = []
for i in range(new_examples):
    lin.fit_sample(x_te[i, :].reshape(1, x.shape[1]), y_te[i])
    online_mae.append(mean_absolute_error(y_te[i+1:], lin.predict(x_te[i+1:, :])))
    
plt.plot(range(len(online_mae)), online_mae)
plt.title('MAE after online learning')
plt.show()

p_vals = lin.p_values
adj_r2 = lin.calculate_stat(x_te, y_te)
print(f'Adjusted R^2 on train set: {lin.adj_r2}; Adjusted R^2 on test set: {adj_r2}')
r2 = lin.r2(x_te, y_te) # normal R^2 on test set