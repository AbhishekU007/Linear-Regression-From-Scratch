import numpy as np

# Custom Linear Regression implementation
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(X.shape[1], dtype=float)
        self.bias = 0.0

        # Gradient descent loop
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Predict using learned weights and bias
        return np.dot(X, self.weights) + self.bias
    
    def mse(self, y_true, y_pred):
        # Mean Squared Error calculation
        return np.mean((y_true - y_pred) ** 2)