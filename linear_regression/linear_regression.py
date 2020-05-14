# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:44:43 2020

@author: Marek
"""

import numpy as np

class LinearRegression():
    def __init__(self, x, y, alpha=0.5, max_iter=100):
        x = np.array(x)
        self.x = np.c_[np.ones((x.shape[0], 1)), x]
        self.y = np.array(y)
        self.alpha = alpha
        self.iter = max_iter
        self.w = np.random.normal(size=(x.shape[1], 1))

if __name__=='__main__':
    x = np.random.normal(size=(500, 5))
    y = np.random.normal(size=(500,1))
    lin = LinearRegression(x, y)