from sklearn.neural_network import MLPClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import os
from evaluate import evaluate_model
from scipy.stats import loguniform, randint

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

# MLP with early stopping
mlp = MLPClassifier(
    random_state=42,
    max_iter=500,
    early_stopping=True,
    n_iter_no_change=10,
    tol=1e-3
)

# Reduced parameter distributions
param_dist = {
    'hidden_layer_sizes': [
        (100,), (100, 100), (100, 100, 50)
    ],
    'activation': ['relu', 'tanh'],
    'alpha': loguniform(1e-4, 1e-2),
    'learning_rate_init': loguniform(1e-4, 1e-1),
    'solver': ['adam']  # Adam is typically faster; drop 'sgd'
}

# Randomized search: 30 samples × 3‑fold CV
rand_search = RandomizedSearchCV(
    estimator=mlp,
    param_distributions=param_dist,
    n_iter=30,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

# Run search
rand_search.fit(X_train, y_train)

print("Best Parameters:", rand_search.best_params_)
print("Best CV Accuracy:", rand_search.best_score_)

# Test-set evaluation
best_mlp = rand_search.best_estimator_
y_pred = best_mlp.predict(X_test)

model_name = "deep_neural_network"
f1_score_weighted = evaluate_model(
    y_test,
    y_pred,
    model_name,
    best_params=rand_search.best_params_,
    save_path=current_dir
)
