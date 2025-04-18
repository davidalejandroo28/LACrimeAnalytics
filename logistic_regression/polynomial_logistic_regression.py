import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from evaluate import evaluate_model
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Path to the preprocessed data.
current_dir = os.path.dirname(os.path.abspath(__file__))

# Compute the path to the project root (which is one level up)
project_root = os.path.dirname(current_dir)

# Construct the absolute path to the data file in the 'data' folder
data_path = os.path.join(project_root, 'data', 'cleaned_crime_data_stratified.csv')
df = pd.read_csv(data_path)

# The target variable is 'crime_code'. All other columns are features.
X = df.drop('crime_code', axis=1)
y = df['crime_code']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initalize the pipeline for polynomial logistic regression.
pipeline = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
])

# Define the grid of hyperparameters.
param_grid = {
    'poly__degree': [2, 3],
    'logreg__C': [0.0001, 0.0003, 0.001, 0.1, 1],
}

# Set up GridSearchCV using the pipeline.
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)

# Evaluate the best model on the test set.
y_pred = grid_search.predict(X_test)
model_name = "logistic_regression_poly"

# Use the evaluation function to generate and save performance reports.
f1_score_weighted = evaluate_model(
    y_test, y_pred, model_name, best_params=grid_search.best_params_, save_path=current_dir
)
