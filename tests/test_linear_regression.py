import numpy as np
from src.linear_regression import LinearRegression

# Test if learned coefficients are close to expected values
def test_coefficients_close():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])  # y = 2x + 1
    model = LinearRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X, y)

    weights = model.weights
    slope = weights[1]
    intercept = weights[0]

    assert abs(slope - 2) < 0.1
    assert abs(intercept - 1) < 0.1

# Test if predictions are close to true values
def test_predictions_close():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])
    model = LinearRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X, y)
    preds = model.predict(X)

    for p, t in zip(preds, y):
        assert abs(p - t) < 0.1
