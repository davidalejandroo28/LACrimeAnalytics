from sklearn.neural_network import MLPClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import os
from evaluate import evaluate_model

# Directory setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Load data
data_path = os.path.join(project_root, 'data', 'cleaned_crime_data_stratified.csv')
df = pd.read_csv(data_path)
print("Dataset preview:")
print(df.head())

# Features / target
X = df.drop('crime_code', axis=1)
y = df['crime_code']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Shallow MLP with early stopping
mlp = MLPClassifier(
    random_state=42,
    max_iter=300,
    early_stopping=True,
    n_iter_no_change=10,
    tol=1e-4
)

# Very small grid: one hidden layer of 50 or 100 units
param_grid = {
    'hidden_layer_sizes': [(50,), (100,)],
    'activation': ['relu'],          # relu is usually fastest
    'alpha': [0.001],                # single regularization strength
    'learning_rate_init': [0.01],    # single learning rate
    'solver': ['adam']               # adam converges quickly
}

# 3‑fold CV → only 2 candidates × 3 folds = 6 fits total
grid_search = GridSearchCV(
    mlp,
    param_grid,
    cv=3,
    n_jobs=-1,
    scoring='accuracy',
    verbose=2
)

# Run the search
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# Test‑set evaluation
best_mlp = grid_search.best_estimator_
y_pred = best_mlp.predict(X_test)

model_name = "shallow_neural_network"
f1_score_weighted = evaluate_model(
    y_test,
    y_pred,
    model_name,
    best_params=grid_search.best_params_,
    save_path=current_dir
)
