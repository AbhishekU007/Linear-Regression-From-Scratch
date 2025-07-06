# Linear Regression from Scratch 🔢

Build, train, and evaluate a linear regression model **without using scikit-learn**. Supports multiple features, batch gradient descent, regularization (Ridge, Lasso, ElasticNet), and a Streamlit UI for interactive predictions and comparison with scikit-learn.

---

## 💡 Features

- Linear Regression from scratch (no ML libraries for core logic)
- Multivariate & univariate regression
- Batch Gradient Descent optimization
- Custom evaluation metrics (MAE, MSE, RMSE, R²)
- Ridge, Lasso, and ElasticNet regularization (from scratch)
- Streamlit web UI for interactive model training and comparison
- Data preprocessing utilities (normalization, standardization, train-test split, add bias)
- Visualizations (predicted vs actual plots)
- Modular structure with unit tests
- Jupyter notebook for EDA, prototyping, and detailed comparison with scikit-learn

---

## 🧠 Tech Stack

- Python
- NumPy, Pandas, Matplotlib
- Streamlit (for web app)
- Pytest (for testing)

---

## 📁 Structure

- `src/`: Core model, metrics, utilities, regularized regression
- `data/`: Input dataset(s)
- `notebooks/`: EDA and prototyping (see `src/linear_reg.ipynb` for a full workflow)
- `tests/`: Unit test cases
- `app.py`: Streamlit web UI

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourname/linear-regression-from-scratch
cd linear-regression-from-scratch
pip install -r requirements.txt
python app.py
```

---

## 📓 Notebooks

- **src/linear_reg.ipynb**: End-to-end notebook with code, markdown explanations, metric comparisons, and regularization experiments.
  - Data loading, preprocessing, and feature engineering
  - Custom vs scikit-learn regression (metrics, plots, similarity)
  - Utility function demos
  - Ridge, Lasso, and ElasticNet from scratch vs scikit-learn
  - Well-commented code and markdown for learning

---

## 🤝 Contributions

Feel free to fork and PR improvements — code, dataset suggestions, tests, or web interface enhancements.

---

## 📚 Author

Notebook and code by [Abhishek Upadhyay](https://github.com/abhisheku007)