# Tips Dataset Regression Analysis

This is my first machine learning project, created to solidify my understanding of Linear Regression. Rather than just applying black-box models, I built the foundational algorithms from scratch to deeply understand the underlying mathematics and mechanics of linear regression, feature selection, and model evaluation—before translating the workflow into a clean, production-grade **scikit-learn** pipeline.

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
- **Q-Q Plots:** Assessing the normality of residuals against theoretical normal quantiles.

### 5. Production Pipeline & Model Serialization
After exploring and proving the math from scratch, I bridged the gap to production machine learning engineering:
- **Scikit-Learn Pipeline:** Combined `ColumnTransformer`, `StandardScaler`, and `SelectFromModel(Lasso)` into an encapsulated, leak-free pipeline.
- **Automated Hyperparameter Tuning:** Used 5-fold cross-validation with `GridSearchCV` to locate the optimal regularization penalty.
- **Post-Lasso Modeling:** Refit an unpenalized `LinearRegression` model on the selected features to eliminate shrinkage bias.
- **Standalone Script & Serialization:** Packaged the workflow into `src/model_training.py` and exported the complete trained pipeline via `joblib` for inference.

---

## 📂 Project Structure

```text
├── notebooks/
│   ├── 1-data-analyze.ipynb                      # Data inspection, missing values, duplicates
│   ├── 2-univariate-data-analysis.ipynb          # Distribution and outlier analysis per feature
│   ├── 3-multivariate-data-analysis.ipynb        # Correlation and relationship exploration
│   ├── 4-feature-selection-by-LassoRegression.ipynb # Custom from-scratch Lasso & alpha tuning
│   ├── 5-feature-selection-by-adjusted-r2.ipynb  # Stepwise feature selection with Adjusted R²
│   ├── 6-log-transformation-comparison.ipynb     # Evaluating log transform vs raw target
│   ├── 7-model-training.ipynb                    # Final manual OLS model & diagnostic plots
│   └── 8-scikit-learn-version.ipynb              # Production sklearn pipeline & GridSearchCV
├── src/
│   └── model_training.py                         # Standalone script to train & export the pipeline
├── model/
│   └── model.joblib                              # Serialized trained pipeline artifact
├── utils.py                                      # Custom math implementations (OLS, metrics)
└── requirements.txt                              # Project dependencies
```

---

## ⚙️ How to Run & Use the Model

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/<your-username>/tips-dataset-regression.git
cd tips-dataset-regression
pip install -r requirements.txt
```

### 2. Train and Export the Model
To retrain the pipeline and export the model artifact to `model/model.joblib`:
```bash
python src/model_training.py
```

### 3. Inference with the Saved Pipeline
Because the entire preprocessing, feature selection, and regression steps are encapsulated in the serialized pipeline, you can feed **raw customer data** directly:

```python
import joblib
import pandas as pd

# Load the trained pipeline
pipeline = joblib.load('model/model.joblib')

# Sample raw customer input
new_customer = pd.DataFrame([{
    'total_bill': 25.50,
    'sex': 'Male',
    'smoker': 'No',
    'day': 'Sun',
    'time': 'Dinner',
    'size': 3
}])

# Predict tip
predicted_tip = pipeline.predict(new_customer)
print(f"Predicted Tip: ${predicted_tip[0]:.2f}")
```

---

## 🛠️ Technologies Used
- **Python 3**
- **NumPy & Pandas:** Data manipulation and custom algorithm implementations.
- **Matplotlib & Seaborn:** Data visualization and statistical diagnostics.
- **Scikit-Learn:** Modern ML pipelines (`ColumnTransformer`, `StandardScaler`, `GridSearchCV`, `SelectFromModel`, `LinearRegression`).
- **SciPy:** Statistical checks and Q-Q plots.
- **Joblib:** Production model serialization.

## 💡 Conclusion
This project served as a comprehensive introduction to machine learning. By first implementing regression math and feature selection from scratch, and subsequently engineering an end-to-end `scikit-learn` pipeline, I gained both a deep theoretical understanding and practical software engineering skills for building production-ready linear models.
