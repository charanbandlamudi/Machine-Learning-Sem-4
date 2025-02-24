# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load the Excel file (Upload manually in Colab)
from google.colab import files
uploaded = files.upload()

# Read the Excel file
file_name = list(uploaded.keys())[0]
excel_data = pd.ExcelFile(file_name)

# Load the 'IRCTC Stock Price' sheet
df = pd.read_excel(file_name, sheet_name='IRCTC Stock Price')

# Data Preprocessing
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna()  # Drop rows with missing values
df['Next_Day_Price'] = df['Price'].shift(-1)  # Next day's price
df['Target'] = (df['Next_Day_Price'] > df['Price']).astype(int)  # 1 if price goes up, else 0
df = df.dropna()  # Drop the last row as it won't have a target

# Select features and target
X = df[['Open', 'High', 'Low', 'Price']]
y = df['Target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Confusion matrices
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

# Classification reports
report_train = classification_report(y_train, y_train_pred)
report_test = classification_report(y_test, y_test_pred)

# Display the results
print("Confusion Matrix - Training Data:")
print(conf_matrix_train)
print("\nClassification Report - Training Data:")
print(report_train)

print("Confusion Matrix - Test Data:")
print(conf_matrix_test)
print("\nClassification Report - Test Data:")
print(report_test)
