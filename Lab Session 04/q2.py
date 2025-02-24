# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np

# Upload the Excel file
from google.colab import files
uploaded = files.upload()

# Read the Excel file
file_name = list(uploaded.keys())[0]
df = pd.ExcelFile(file_name)

# Load the 'IRCTC Stock Price' sheet
df_irctc = pd.read_excel(file_name, sheet_name='IRCTC Stock Price')

# Prepare the data for price prediction
df_irctc = df_irctc[['Price', 'Open', 'High', 'Low']].dropna()

# Convert columns to numeric
df_irctc['Price'] = pd.to_numeric(df_irctc['Price'], errors='coerce')
df_irctc['Open'] = pd.to_numeric(df_irctc['Open'], errors='coerce')
df_irctc['High'] = pd.to_numeric(df_irctc['High'], errors='coerce')
df_irctc['Low'] = pd.to_numeric(df_irctc['Low'], errors='coerce')

# Drop rows with NaN values
df_irctc = df_irctc.dropna()

# Features and target
X = df_irctc[['Open', 'High', 'Low']]
y = df_irctc['Price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Predictions
y_train_pred = reg_model.predict(X_train)
y_test_pred = reg_model.predict(X_test)

# Calculate metrics for training data
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Calculate metrics for test data
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Display the results
print("Training Data Metrics:")
print(f"MSE: {mse_train}")
print(f"RMSE: {rmse_train}")
print(f"MAPE: {mape_train}")
print(f"R² Score: {r2_train}\n")

print("Test Data Metrics:")
print(f"MSE: {mse_test}")
print(f"RMSE: {rmse_test}")
print(f"MAPE: {mape_test}")
print(f"R² Score: {r2_test}")
