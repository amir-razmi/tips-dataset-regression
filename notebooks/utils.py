import numpy as np

# Linear Regression Algorithms:
def ols(X, y):
  return np.linalg.inv(X.T @ X) @ X.T @ y

def lasso_regression(X, y, learning_rate=0.01, n_iterations=1000, alpha=0.01):
  y = y.reshape(-1, 1)
  n = y.shape[0]
  p = X.shape[1]
  B = np.zeros(p).reshape(-1, 1)
  for i in range(n_iterations):
    gradient = 1/n * X.T @ (X@B - y) + alpha * np.sign(B)
    B -= learning_rate * gradient
  return B.reshape(-1)


#--------------------------------------------
# Evaluation Metrics:

def mse(y, pred):
  return np.mean((y - pred) ** 2)

def rmse(y, pred):
  return np.sqrt(mse(y, pred))

def mae(y, pred):
  return np.mean(np.abs(y - pred))

def r_squared(y, pred):
  mean_y = np.mean(y)
  return 1 - (np.sum((y - pred) ** 2) / np.sum((y - mean_y) ** 2))

def adjusted_r_squared(y, pred, n_features):
  return 1 - (
    ((1 - r_squared(y, pred)) * (y.shape[0] - 1)) / 
    (y.shape[0] - n_features - 1)
  )
