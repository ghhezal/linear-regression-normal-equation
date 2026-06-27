import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

class linearRegressionNormal:

    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        Xb = np.c_[np.ones((X.shape[0], 1)), X]

        A = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
        self.intercept_ = A[0]
        self.coef_ = A[1 :]

    def predict(self, X):
        X = np.array(X)

        return X @ self.coef_ + self.intercept_
    

reg = linearRegressionNormal()

X, y = fetch_california_housing(return_X_y=True)

reg.fit(X, y)

y_pred = reg.predict(X)

print(r2_score(y, y_pred))

reg_sk = LinearRegression()

reg_sk.fit(X, y)

y_pred = reg_sk.predict(X)

print(r2_score(y, y_pred))