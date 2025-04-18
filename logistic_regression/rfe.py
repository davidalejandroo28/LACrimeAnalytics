import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline

from evaluate import evaluate_model

# Determine paths.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, 'data', 'cleaned_crime_data_stratified.csv')

# Load the preprocessed dataset.
df = pd.read_csv(data_path)
print("Dataset preview:")
print(df.head())

# Separate features and target.
X = df.drop('crime_code', axis=1)
y = df['crime_code']

# Split the data into training and testing sets (using stratification).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize the base estimator for feature elimination.
base_estimator = LogisticRegression(max_iter=1000, random_state=42)

# Set up RFECV for backward elimination.
rfecv = RFECV(estimator=base_estimator, step=1, cv=5, scoring='f1_weighted')

# Set up the logistic regression model with the fixed hyperparameters.
logistic_regression = LogisticRegression(C=1, class_weight=None, solver='lbfgs', max_iter=1000, random_state=42)

# Create a pipeline: first perform backward feature elimination, then fit logistic regression.
pipeline = Pipeline([
    ('feature_selection', rfecv),
    ('logreg', logistic_regression)
])

# Fit the pipeline on the training data.
pipeline.fit(X_train, y_train)

# See how many features are selected.
selected_features = pipeline.named_steps['feature_selection'].n_features_
print(f"Number of features selected by backward elimination: {selected_features}")

# Evaluate the model on the test set.
y_pred = pipeline.predict(X_test)
model_name = "logistic_regression_rfecv"

f1_score_weighted = evaluate_model(
    y_test, y_pred, model_name, best_params={'C': 1, 'class_weight': None, 'solver': 'lbfgs'}, save_path=current_dir
)
