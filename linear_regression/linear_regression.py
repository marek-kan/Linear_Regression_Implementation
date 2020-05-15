# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:44:43 2020

@author: Marek
"""

import numpy as np

class LinearRegression():
    def __init__(self, learning_rate = 0.1, beta=0.9, reg_lambda=0.5, max_iter=100):
        self.reg_lambda = reg_lambda
        self.iter = max_iter
        self.lr = learning_rate
        self.b = beta
        
    def cost(self, x, y, m):
        c = 1/(2*m) * (x @ self.w - y)**2
        regC = c + 1/(2*m) * self.reg_lambda * self.w.T @ self.w**2
        return sum(regC)
    
    def update(self, x, y, m):
        pred = x @ self.w
        grad0 = 1/m * (x[:, 0].T @ (pred - y))
        grad1 = 1/m * (x[:, 1:].T @ (pred - y))
        # compute accelerations for bias and coefficients
        self.v[0] = self.b*self.v[0] + (1-self.b)*grad0
        self.v[1:] = self.b*self.v[1:] + (1-self.b)*grad1
        # Updating the weights
        self.w[0] = self.w[0] - self.lr * self.v[0]
        self.w[1:] = self.w[1:]*(1 - self.lr * self.reg_lambda/m) - self.lr * self.v[1:]
        
    def fit(self, x, y):
        x = np.array(x)
        m = len(y)
        x = np.c_[np.ones((m, 1)), x]
        y = np.array(y).reshape(m, 1)
        self.w = np.random.normal(size=(x.shape[1], 1))
        self.v = np.zeros(shape=(x.shape[1], 1)) # initialize momentum acceleration to zeros
        self.costs = []
        for i in range(self.iter):
            self.update(x, y, m)
            self.costs.append(self.cost(x, y, m)[0])
        return print(f'Final cost {self.costs[-1]}')
    
    def fit_sample(self, x, y, iterations=1):
        """
        Intended usage: During online learning when model is initialized by .fit()
        and .fit_sample() used in online production.
        """
        x = np.array(x)
        try:
            m = len(y)
        except:
            m = 1
        x = np.c_[np.ones((m, 1)), x]
        y = np.array(y).reshape(m, 1)
        for i in range(iterations):
            self.update(x, y, m)
            self.costs.append(self.cost(x, y, m)[0])
        return print(f'Last loss: {self.costs[-2]}; New loss: {self.costs[-1]}')
       
    def predict(self, x):
        x = np.array(x)
        x = np.c_[np.ones((x.shape[0])), x]
        return x @ self.w


    
    
    