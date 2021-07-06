import numpy as np
import xlsxwriter.shape


class LinearRegression:

    def __init__(self, alpha=0.01, num_iters=1000, weight_init=None, bias_init=None):
        self.alpha = alpha
        self.num_iters = num_iters

    def init_params(self, weight_init, bias_init=0):
        self.bias = bias_init
        self.weights = weight_init

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.init_params(np.zeros(num_features), bias_init=self.bias)
        for _ in range(num_samples):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)
            self.weights -= dw * self.alpha
            self.bias -= db * self.alpha
