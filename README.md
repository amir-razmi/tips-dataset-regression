# Tips Dataset Regression Analysis

This is my first machine learning project, created to solidify my understanding of Linear Regression. Rather than just applying black-box models, I built the foundational algorithms from scratch to deeply understand the underlying mathematics and mechanics of linear regression, feature selection, and model evaluation.

## 🎯 Project Objective
The goal of this project is to predict the **tip amount** given at a restaurant based on various dining features (total bill, sex, smoker, day, time, size) using the classic `tips` dataset from Seaborn.

## 🚀 Key Highlights & Learnings

### 1. Algorithms from Scratch
To truly grasp the mechanics of linear regression, I implemented the Ordinary Least Squares (OLS) weights mathematically from scratch using NumPy (`np.linalg.inv(X.T @ X) @ X.T @ y`). I also wrote custom implementations for evaluation metrics such as MSE, RMSE, MAE, R², and Adjusted R².

### 2. Triangulated Feature Selection
I performed feature selection in three distinct ways to practice different techniques and compare their outcomes:
- **Visual Exploratory Data Analysis (EDA):** Identifying relationships and correlations through data visualization.
- **Lasso Regression (L1 Regularization):** Using penalty terms to shrink less important feature coefficients to zero.
- **Adjusted R² Evaluation:** Iteratively evaluating features to see the impact on model performance while penalizing for complexity.

Through this triangulation, I observed consistent results across methods, ultimately confirming which features (like `size`) to drop based on cross-validated evidence.

### 3. Model Diagnostics & Validation
A model is only as good as its assumptions. I evaluated the final model using robust statistical diagnostics:
- **Actual vs. Fitted Plots:** Visualizing the model's predictions against real data on both training and test sets.
- **Residual Analysis:** Plotting residuals vs. predicted values to check for homoscedasticity (constant variance of errors) and using KDE plots to verify the normal distribution of residuals.

## 📂 Project Structure

The project is structured into a series of Jupyter Notebooks that document my step-by-step workflow:

- `1-data-analyze.ipynb`: Initial data loading, cleaning, and understanding.
- `2-data-preprocessing.ipynb`: Handling missing values, outliers, and encoding categorical variables.
- `3-multivariate-data-analysis.ipynb`: Deep dive into Exploratory Data Analysis (EDA) and visualization.
- `4-feature-selection-by-LassoRegression.ipynb`: Feature selection using L1 regularization.
- `5-feature-selection-by-adjusted-r2.ipynb`: Feature selection using Adjusted R² metrics.
- `6-model-training.ipynb`: Final model training, evaluation, and residual diagnostics.

## 🛠️ Technologies Used
- **Python 3**
- **NumPy & Pandas:** Data manipulation and custom algorithm implementation.
- **Matplotlib & Seaborn:** Data visualization and model diagnostics.
- **Scikit-Learn:** Data splitting and Lasso Regression.

## 💡 Conclusion
This project served as a comprehensive introduction to machine learning. By manually implementing regression math and methodically testing different feature selection techniques, I gained a strong, intuitive understanding of how linear models work, how to evaluate them, and why checking statistical assumptions is crucial.
