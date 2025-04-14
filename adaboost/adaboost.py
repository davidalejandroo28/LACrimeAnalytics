from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from evaluate import evaluate_model
import os

# Path to the preprocessed data.
current_dir = os.path.dirname(os.path.abspath(__file__))

# Compute the path to the project root (which is one level up)
project_root = os.path.dirname(current_dir)

# Construct the absolute path to the data file in the 'data' folder
data_path = os.path.join(project_root, 'data', 'cleaned_crime_data_stratified.csv')
df = pd.read_csv(data_path)
print("Dataset preview:")
print(df.head())

# The target variable is 'crime_code'. All other columns are features.
X = df.drop('crime_code', axis=1)
y = df['crime_code']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

base_estimator = DecisionTreeClassifier(max_depth=1)

# Create the AdaBoost classifier with the specified base estimator.
ada = AdaBoostClassifier(estimator=base_estimator, random_state=42)

# Define the parameter grid to search through.
param_grid = {
    'n_estimators': [50, 100, 200, 500, 1000],
    'learning_rate': [0.01, 0.03, 0.1, 0.3, 1.0],
    'estimator__max_depth': [1, 2, 3]
}

# Create the GridSearchCV object with 5-fold cross-validation.
grid_search = GridSearchCV(ada, param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Fit the grid search to the training data.
grid_search.fit(X_train, y_train)

# Print the best parameters and best score from the grid search.
print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# Evaluate the best estimator on the test set.
best_ada = grid_search.best_estimator_
y_pred = best_ada.predict(X_test)

model_name = "adaboost"

# Evaluate the model and save the report.
f1_score_weighted = evaluate_model(
    y_test, y_pred, model_name, best_params=grid_search.best_params_, save_path=current_dir
)