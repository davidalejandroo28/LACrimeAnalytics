import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from evaluate import evaluate_model
import os

# Path to the preprocessed data.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, 'data', 'cleaned_crime_data_stratified.csv')

# Load dataset
df = pd.read_csv(data_path)

# Split features and target
X = df.drop('crime_code', axis=1)
y = df['crime_code']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize the SVM model
model = SVC(random_state=42)

# Define hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'class_weight': ['balanced', None]
}

# Grid search
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)

# Predictions and evaluation
y_pred = grid_search.predict(X_test)
model_name = "svm"

f1_score_weighted = evaluate_model(
    y_test, y_pred, model_name, best_params=grid_search.best_params_, save_path=current_dir
)
