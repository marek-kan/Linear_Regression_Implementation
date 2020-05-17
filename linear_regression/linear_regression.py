# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:44:43 2020

@author: Marek
"""

import numpy as np
import scipy.stats as stat

class LinearRegression():
    def __init__(self, learning_rate = 0.1, beta=0.9, reg_lambda=0.5, max_iter=100):
        self.reg_lambda = reg_lambda
        self.iter = max_iter
        self.lr = learning_rate
        self.b = beta
        
    def cost(self, x, y, m):
        c = 1/(2*m) * (np.dot(x, self.w) - y)**2
        regC = c + 1/(2*m) * self.reg_lambda * np.dot(self.w.T, self.w**2)
        return sum(regC)
    
    def update(self, x, y, m):
        pred = np.dot(x, self.w)
        grad0 = 1/m * (np.dot(x[:, 0].T, (pred - y)))
        grad1 = 1/m * (np.dot(x[:, 1:].T, (pred - y)))
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
        self.calculate_stat(x, y, training=True)
        return print(f'Final loss {self.costs[-1]}')
    
    def fit_sample(self, x, y, iterations=1):
        """
        Intended usage: During online learning after model is initialized by .fit().
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
        return np.dot(x, self.w)
    
    def r2(self, x, y, training=False):
        """
        training=False, calculate on test set
        """
        if not training:
            pred = self.predict(x)
            y = np.array(y)
            y = y.reshape(y.shape[0], 1)
        else:
            pred = self.predict(x[:, 1:])
        # model variance
        mod_var = sum((pred - y)**2)
        # total variance
        tot_var = sum((y.mean() - y)**2)
        # variance explained by model
        exp_var = tot_var - mod_var
        return (exp_var/tot_var)[0]
        
    
    def calculate_stat(self, x, y, training=False):
        """
        training=False, calculate on test set
        """
        if not training:
            pred = self.predict(x)
            x = np.array(x)
            x = np.c_[np.ones((x.shape[0])), x]
            y = np.array(y)
            y = y.reshape(y.shape[0], 1)
        else:
            pred = self.predict(x[:, 1:])
        # Calculate SSE (sum of squared errors) and SE (standard error)
        sse = np.sum((pred - y) ** 2) / float(x.shape[0] - x.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(x.T, x))))])
        # Compute the t-statistic
        self.t_stat = self.w.reshape(self.w.shape[0]) / se
        # Find the p-value for each feature
        self.p_values = np.squeeze(2 * (1 - stat.t.cdf(np.abs(self.t_stat), y.shape[0] - x.shape[1])))
        #adjusted R^2, True because I added ones before
        if not training:
            # only return adjusted R^2 on test set
            adj_score = (1-(1-self.r2(x, y, True))*(x.shape[0]-1)/(x.shape[0]-x.shape[1]-1))
            return adj_score
        else:
            self.adj_score = (1-(1-self.r2(x, y, True))*(x.shape[0]-1)/(x.shape[0]-x.shape[1]-1))
        


    
    
    