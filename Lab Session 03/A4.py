import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Upload and load the dataset
from google.colab import files
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
xls = pd.ExcelFile(file_name)

# Load the "IRCTC Stock Price" sheet
df = pd.read_excel(xls, sheet_name="IRCTC Stock Price")

# Select relevant features (e.g., 'Open' and 'Low' prices as predictors, 'High' as target)
X = df[['Open', 'Low']]
y = df['High']

# Ensure features are numeric
X = X.select_dtypes(include=[np.number])
y = pd.to_numeric(y, errors='coerce')

# Drop any NaN values
df = df.dropna(subset=['Open', 'Low', 'High'])

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train kNN regressor with k=3
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate the model
score = knn.score(X_test, y_test)
print("Model R² Score:", score)

knn.fit(X_train, y_train)

score = knn.score(X_test, y_test)
print("Model R² Score:", score)
