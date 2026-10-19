import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

def train_and_save_model():
    seed = 0
    df = sns.load_dataset('tips')

    X, y = df.loc[:, df.columns != 'tip'], df['tip']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


    encoder = ColumnTransformer(
      transformers=[
        ('cat', OneHotEncoder(drop='if_binary'), ['sex', 'smoker', 'time', 'day']),
      ],
      remainder='passthrough'
    )

    pipeline = Pipeline(
      steps=[
        ("encode", encoder),
        (
          "feature_selection", 
          SelectFromModel(estimator=Lasso(alpha=0.14, random_state=seed))
        ),
        ("regression", LinearRegression())
      ],
    )
    # 1. Fit the entire pipeline
    pipeline.fit(X_train, y_train)

    # 2. Evaluate on test set
    y_pred = pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = root_mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    print("Model Training Complete!")
    print(f"Test R²:   {test_r2:.4f}")
    print(f"Test RMSE: ${test_rmse:.2f}")
    print(f"Test MAE:  ${test_mae:.2f}")

    # 3. Ensure the output directory exists and save the FULL pipeline
    os.makedirs('model', exist_ok=True)
    model_path = 'model/model.joblib'
    joblib.dump(pipeline, model_path)
    print(f"Pipeline successfully saved to {model_path}")

if __name__ == '__main__':
    train_and_save_model()