# Linear Regression Implementation
Implementation of linear regression on steriods.

Optimized by Batch Gradient Descent with Momentum. I add possibility to fit another sample into trained estimator. All this is done in Python with help of NumPy.

I compare sklearn's Ridge vs my Linear Regression in test.py on boston housing dataset. Results (in MAE) are:
 * Sci-kit Learn: 3.3616
 * My LR: 3.3242
 * My LR after 5 "online" examples: 3.1300

I find online learning possibility useful because there is no need for cyclic offline retraining and deployment. Instead, after obtaining true value of target in production it can be fitted to the estimator without need of goinig offline.
