import streamlit as st
import pandas as pd
import numpy as np

from src.linear_regression import LinearRegression
from src.metrics import mean_squared_error as mse, mean_absolute_error as mae, root_mean_squared_error as rmse, r2_score
from src.utils import train_test_split, normalize, standardize, plot_predictions, add_bias
from src.regularized_regression import RidgeRegression, LassoRegression, ElasticNetRegression

from sklearn.linear_model import LinearRegression as SklearnLR, Ridge as SklearnRidge, Lasso as SklearnLasso, ElasticNet as SklearnElasticNet

st.title("Regression Playground (Custom vs Scikit-learn)")

file = st.file_uploader("Upload CSV", type=['csv'])

if file:
    df = pd.read_csv(file)
    if df.isnull().values.any():
        st.warning("NaN values detected in your dataset. They will be dropped for regression to work.")
        df = df.dropna()
    st.write("Dataset Preview:", df.head())

    target = st.selectbox("Select Target Variable", df.columns)
    features = st.multiselect("Select Feature Columns", df.columns.drop(target))

    model_choice = st.selectbox("Choose Model", ["Linear", "Ridge", "Lasso", "Elastic Net"])
    test_size = st.slider("Test/Train Split Ratio", 0.1, 0.5, 0.2)

    # Show regularization sliders only if needed
    alpha = None
    l1_ratio = None
    if model_choice == "Ridge":
        alpha = st.slider("Alpha (L2 penalty)", 0.0, 1.0, 0.1)
    elif model_choice == "Lasso":
        alpha = st.slider("Alpha (L1 penalty)", 0.0, 1.0, 0.1)
    elif model_choice == "Elastic Net":
        alpha = st.slider("Alpha (L1+L2 penalty)", 0.0, 1.0, 0.1)
        l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5)

    if st.button("Train Model") and features and target:
        X = df[features].values
        y = df[target].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        X_train = standardize(X_train)
        X_test = standardize(X_test)

        # For custom models, add bias if needed
        X_train_bias = add_bias(X_train)
        X_test_bias = add_bias(X_test)

        # Model selection and training
        if model_choice == "Linear":
            # Custom
            custom_model = LinearRegression(learning_rate=0.01, n_iterations=5000)
            custom_model.fit(X_train, y_train)
            y_pred_custom = custom_model.predict(X_test)
            # Sklearn
            skl_model = SklearnLR()
            skl_model.fit(X_train, y_train)
            y_pred_skl = skl_model.predict(X_test)
        elif model_choice == "Ridge":
            custom_model = RidgeRegression(learning_rate=0.01, n_iters=1000, alpha=alpha)
            custom_model.fit(X_train_bias, y_train)
            y_pred_custom = custom_model.predict(X_test_bias)
            skl_model = SklearnRidge(alpha=alpha, fit_intercept=True, max_iter=1000)
            skl_model.fit(X_train, y_train)
            y_pred_skl = skl_model.predict(X_test)
        elif model_choice == "Lasso":
            custom_model = LassoRegression(alpha=alpha, n_iters=1000)
            custom_model.fit(X_train_bias, y_train)
            y_pred_custom = custom_model.predict(X_test_bias)
            skl_model = SklearnLasso(alpha=alpha, fit_intercept=True, max_iter=1000)
            skl_model.fit(X_train, y_train)
            y_pred_skl = skl_model.predict(X_test)
        else:  # Elastic Net
            custom_model = ElasticNetRegression(learning_rate=0.01, n_iters=1000, alpha=alpha, l1_ratio=l1_ratio)
            custom_model.fit(X_train_bias, y_train)
            y_pred_custom = custom_model.predict(X_test_bias)
            skl_model = SklearnElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=1000)
            skl_model.fit(X_train, y_train)
            y_pred_skl = skl_model.predict(X_test)

        st.subheader("Custom Model: Prediction vs Actual")
        st.pyplot(plot_predictions(y_test, y_pred_custom))

        st.subheader("Scikit-learn Model: Prediction vs Actual")
        st.pyplot(plot_predictions(y_test, y_pred_skl))

        st.subheader("Custom Model Metrics")
        st.write("MSE:", mse(y_test, y_pred_custom))
        st.write("MAE:", mae(y_test, y_pred_custom))
        st.write("RMSE:", rmse(y_test, y_pred_custom))
        st.write("R2 Score:", r2_score(y_test, y_pred_custom))

        st.subheader("Scikit-learn Model Metrics")
        st.write("MSE:", mse(y_test, y_pred_skl))
        st.write("MAE:", mae(y_test, y_pred_skl))
        st.write("RMSE:", rmse(y_test, y_pred_skl))
        st.write("R2 Score:", r2_score(y_test, y_pred_skl))

        # Similarity
        similarity = 100 - (np.mean(np.abs(y_pred_custom - y_pred_skl)) / np.mean(np.abs(y_pred_skl)) * 100)
        st.write(f"Prediction similarity to scikit-learn: {similarity:.2f}%")
    elif not features or not target:
        st.warning("Please select both features and target variable to train the model.")