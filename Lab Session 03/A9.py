# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Upload and load the dataset
from google.colab import files
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
xls = pd.ExcelFile(file_name)

# Load the "IRCTC Stock Price" sheet
df = pd.read_excel(xls, sheet_name="IRCTC Stock Price")

# Check if 'Close' column exists, if not, use 'Price' or 'High' instead
if 'Close' not in df.columns:
    # Assuming 'Price' or 'High' represents the closing price
    closing_price_col = 'Price'  # Change to 'High' if appropriate
    print(f"Warning: 'Close' column not found. Using '{closing_price_col}' instead.")
else:
    closing_price_col = 'Close'

# Assuming the dataset has a categorical target variable for classification
# Convert the target variable to categorical labels (e.g., 'High' stock movement: 1, 'Low': 0)
df['Trend'] = (df[closing_price_col] > df['Open']).astype(int)  # Example binary classification target
X = df[['Open', 'Low']]
y = df['Trend']

# Ensure features are numeric
X = X.select_dtypes(include=[np.number])
y = pd.to_numeric(y, errors='coerce')

# Drop any NaN values
df = df.dropna(subset=['Open', 'Low', closing_price_col]) # Update dropna to use closing_price_col

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train kNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on test set
y_pred = knn.predict(X_test)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Compute classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
