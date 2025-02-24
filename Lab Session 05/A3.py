import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

numerical_columns = ["Start time", "End time"]
X = df_combined[numerical_columns].values

y = df_combined["Label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_all_features = LinearRegression().fit(X_train, y_train)

# Predictions
y_train_pred = model_all_features.predict(X_train)
y_test_pred = model_all_features.predict(X_test)

# Output metrics
print("Train Set Metrics (All Features):", compute_metrics(y_train, y_train_pred))
print(" Test Set Metrics (All Features):", compute_metrics(y_test, y_test_pred))
