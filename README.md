# Linear Regression Implementation
Implementation of linear regression on steriods.

Optimized by Gradient Descent with Momentum. I add possibility to fit another sample into trained estimator. All this is done in Python with help of NumPy.

I compare sklearn's Ridge vs my Linear Regression in test.py on boston housing dataset. Results (in MAE) are about the same, my LR (after initial training) had lower error on test set by 0.0006. After 5 examples learned "online", my LR scored on the rest of the test set better by ~0.087 than initial Ridge regressor.

I find this useful because there is no need for cyclic offline retraining and deployment. Instead, after obtaining true value of target in production it can be fitted to the estimator without need of goinig offline.
