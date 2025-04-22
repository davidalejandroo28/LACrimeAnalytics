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
param_grid = {
    'n_estimators': [200, 300, 500],         
    'max_depth': [10, 15],                   
    'min_samples_split': [5, 10],            
    'min_samples_leaf': [1, 2],              
    'max_features': ['sqrt', 'log2'],        
    'bootstrap': [True],                   
    'class_weight': ['balanced']             
}


# Run grid search
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Predict
y_pred = grid_search.predict(X_test)

# Evaluate
# Evaluate model using external function
f1 = evaluate_model(
    eval_y=y_test,
    pred_y=y_pred,
    model_name="RandomForest",
    best_params=grid_search.best_params_,
    save_path=current_dir
)
