import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Import the evaluation function.
from evaluate import evaluate_model

# Determine directory paths.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, 'data', 'cleaned_crime_data_stratified.csv')

# Load the preprocessed dataset.
df = pd.read_csv(data_path)
print("Dataset preview:")
print(df.head())

# The target variable is 'crime_code'. All other columns are features.
X = df.drop('crime_code', axis=1)
y = df['crime_code']

# Split the data into training and testing sets (stratified by the target variable).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define the Decision Tree classifier.
model = DecisionTreeClassifier(random_state=42)

# Define the parameter grid for GridSearchCV.
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Set up GridSearchCV.
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)

# Evaluate the model on the test set.
y_pred = grid_search.predict(X_test)
model_name = "decision_tree"

# Use the evaluation function to generate and save performance reports.
f1_score_weighted = evaluate_model(
    y_test, y_pred, model_name, best_params=grid_search.best_params_, save_path=current_dir
)
