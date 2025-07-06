app.py is a **Streamlit web application** that lets users interactively compare custom regression models (implemented from scratch) with scikit-learn’s regression models. Here’s how it works:

---

### 1. **Imports**
- Loads Streamlit, pandas, numpy, and all custom and scikit-learn regression/model utility functions.

### 2. **App Title**
- Sets the app title: **"Regression Playground (Custom vs Scikit-learn)"**

### 3. **File Upload**
- Users can upload a CSV dataset via the sidebar.

### 4. **Dataset Preview**
- Shows the first few rows of the uploaded dataset.

### 5. **Feature & Target Selection**
- Users select the target variable and input features from dropdowns in the sidebar.

### 6. **Model Selection**
- Users choose which regression model to use:  
  - Linear Regression  
  - Ridge Regression  
  - Lasso Regression  
  - Elastic Net Regression

### 7. **Test Size**
- Users select the test/train split ratio.

### 8. **Model Training**
- When the "Train Model" button is pressed:
  - The data is split into train/test sets.
  - Features are standardized.
  - For custom models, a bias column is added.
  - The selected model (custom and scikit-learn) is trained and used to predict on the test set.
  - For Ridge, Lasso, and Elastic Net, users can adjust regularization parameters (`alpha`, `l1_ratio`).

### 9. **Results & Visualization**
- **Prediction vs Actual** plots for both custom and scikit-learn models.
- **Metrics** for both models:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R2 Score
- **Prediction similarity** percentage between custom and scikit-learn models.

### 10. **User Guidance**
- If features or target are not selected, a warning is shown.

---

**In summary:**  
This app provides an interactive playground for regression, allowing users to upload data, select models and parameters, and visually and numerically compare custom and scikit-learn regression results.