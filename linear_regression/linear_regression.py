# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:44:43 2020

@author: Marek
"""

import numpy as np

class LinearRegression():
    def __init__(self, x, y, learning_rate = 0.1, reg_lambda=0.5, max_iter=100):
        x = np.array(x)
        self.m = len(y)
        self.x = np.c_[np.ones((self.m, 1)), x]
        self.y = np.array(y)
        self.reg_lambda = reg_lambda
        self.iter = max_iter
        self.lr = learning_rate
        self.w = np.random.normal(size=(self.x.shape[1], 1))
        
    def cost(self):
        c = 1/(2*self.m) * (self.x @ self.w - self.y)**2
        regC = c + 1/(2*self.m) * self.reg_lambda * self.w.transpose() @ self.w
        return sum(regC)
    
    def update(self):
        pred = self.x @ self.w
        grad0 = self.lr * 1/self.m * sum(pred - self.y)*self.w[0]
        grad1 = self.lr * 1/self.m * sum(pred - self.y)*self.w[1:]
        self.w[0] = self.w[0] - grad0
        self.w[1:] = self.w[1:]*(1 - self.lr * self.reg_lambda/self.m) - grad1

if __name__=='__main__':
    x = np.random.normal(size=(500, 5))
    y = np.random.normal(size=(500,1))
    lin = LinearRegression(x, y)