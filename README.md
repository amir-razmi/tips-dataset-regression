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

### 3. Hypothesis Testing (Log Transformation)
I noticed heteroscedasticity (increasing variance) and positive skew in the target variable. To address this, I ran an experiment comparing a standard linear model to a log-transformed model side-by-side, analyzing the metrics and residual distributions to make an informed, data-driven decision on which model to proceed with.

### 4. Model Diagnostics & Validation
A model is only as good as its assumptions. I evaluated the final model using robust statistical diagnostics:
- **Actual vs. Fitted Plots:** Visualizing the model's predictions against real data on both training and test sets.
- **Residual Analysis:** Plotting residuals vs. predicted values to check for homoscedasticity (constant variance of errors) and using KDE plots to verify the normal distribution of residuals.

## 📂 Project Structure

The project is structured into a series of Jupyter Notebooks that document my step-by-step workflow:

- `1-data-analyze.ipynb`: Initial data loading, inspection, and checking for missing values or duplicates.
- `2-univariate-data-analysis.ipynb`: Analyzing features individually to understand their shapes, structures, and overall distributions.
- `3-multivariate-data-analysis.ipynb`: Exploring relationships between multiple features and the target variable to guide feature selection.
- `4-feature-selection-by-LassoRegression.ipynb`: Using a custom Lasso Regression algorithm and hyperparameter tuning to filter out useless features.
- `5-feature-selection-by-adjusted-r2.ipynb`: Iteratively evaluating features using Adjusted R² to mathematically confirm which features add predictive power.
- `6-log-transformation-comparison.ipynb`: Comparing a regular linear model against a log-transformed model to address heteroscedasticity and positive skew.
- `7-model-training.ipynb`: Training the final Ordinary Least Squares (OLS) regression model and performing rigorous residual analysis (Q-Q plots, KDE).

## 🛠️ Technologies Used
- **Python 3**
- **NumPy & Pandas:** Data manipulation and custom algorithm implementation.
- **Matplotlib & Seaborn:** Data visualization and model diagnostics.
- **Scikit-Learn:** Data splitting and Lasso Regression.

## 💡 Conclusion
This project served as a comprehensive introduction to machine learning. By manually implementing regression math and methodically testing different feature selection techniques, I gained a strong, intuitive understanding of how linear models work, how to evaluate them, and why checking statistical assumptions is crucial.
