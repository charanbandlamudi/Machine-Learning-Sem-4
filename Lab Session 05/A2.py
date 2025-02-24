from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np

# Predictions
y_train_pred = model_one_feature.predict(X_train)
y_test_pred = model_one_feature.predict(X_test)

# Compute metrics
def compute_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2 Score": r2_score(y_true, y_pred)
    }

print("Train Set Metrics:", compute_metrics(y_train, y_train_pred))
print( "Test Set Metrics:", compute_metrics(y_test, y_test_pred))
