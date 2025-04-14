import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import importlib.util
import os

# Get the absolute path to your local evaluate.py
current_dir = os.getcwd()
project_root = os.path.dirname(current_dir)
evaluate_path = os.path.join(project_root, 'evaluate.py')

# Dynamically load evaluate_model from evaluate.py
spec = importlib.util.spec_from_file_location("evaluate", evaluate_path)
evaluate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_module)

evaluate_model = evaluate_module.evaluate_model

import os

current_dir = os.getcwd()
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, "data", "cleaned_crime_data_stratified.csv")
# Load data
df = pd.read_csv(data_path)
X = df.drop('crime_code', axis=1)
y = df['crime_code']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define model and grid search params
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}

# Run grid search
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Predict
y_pred = grid_search.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Output
print(f"\nModel: random_forest")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
print("\nBest Hyperparameters:")
print(grid_search.best_params_)
