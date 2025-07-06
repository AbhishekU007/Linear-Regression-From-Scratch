import numpy as np

# Lasso Regression using coordinate descent
class LassoRegression:
    def __init__(self, alpha=1.0, n_iters=1000, tol=1e-4):
        self.alpha = alpha
        self.n_iters = n_iters
        self.tol = tol
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        # Coordinate descent optimization
        for _ in range(self.n_iters):
            for j in range(n_features):
                y_pred = np.dot(X, self.weights)
                residual = y - y_pred + self.weights[j] * X[:, j]

                rho = np.dot(X[:, j], residual)

                # Soft thresholding
                if rho < -self.alpha / 2:
                    self.weights[j] = (rho + self.alpha / 2) / np.dot(X[:, j], X[:, j])
                elif rho > self.alpha / 2:
                    self.weights[j] = (rho - self.alpha / 2) / np.dot(X[:, j], X[:, j])
                else:
                    self.weights[j] = 0

    def predict(self, X):
        return np.dot(X, self.weights)

# Ridge Regression using gradient descent
class RidgeRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, alpha=1.0):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.alpha = alpha  # regularization strength
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(X.shape[1])

        # Gradient descent optimization
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights)
            gradient = (2 / n_samples) * (np.dot(X.T, (y_pred - y)) + self.alpha * self.weights)
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        return np.dot(X, self.weights)

# Elastic Net Regression (mix of L1 and L2)
class ElasticNetRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, alpha=1.0, l1_ratio=0.5):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.alpha = alpha  # Regularization strength
        self.l1_ratio = l1_ratio  # Mix between L1 and L2 (0 = L2 only, 1 = L1 only)
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent with combined L1 and L2 penalty
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (-2 / n_samples) * np.dot(X.T, (y - y_pred)) + self.alpha * (
                self.l1_ratio * np.sign(self.weights) + (1 - self.l1_ratio) * self.weights
            )
            db = (-2 / n_samples) * np.sum(y - y_pred)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
