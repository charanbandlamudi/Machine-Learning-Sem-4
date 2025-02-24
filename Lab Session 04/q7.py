# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Upload the Excel file
from google.colab import files
uploaded = files.upload()

# Read the Excel file
file_name = list(uploaded.keys())[0]
df = pd.ExcelFile(file_name)

# Load the 'IRCTC Stock Price' sheet
df_irctc = pd.read_excel(file_name, sheet_name='IRCTC Stock Price')

# Prepare the data
# Using 'Open' as X and 'Price' as Y for classification
df_irctc = df_irctc[['Open', 'Price']].dropna()

# Convert columns to numeric
df_irctc['Open'] = pd.to_numeric(df_irctc['Open'], errors='coerce')
df_irctc['Price'] = pd.to_numeric(df_irctc['Price'], errors='coerce')

# Drop rows with NaN values
df_irctc = df_irctc.dropna()

# Take first 100 data points for training and testing
X = df_irctc[['Open', 'Price']].values[:100]

# Class Labels:
# Class 0 (Blue) -> Price < Open (Stock went down)
# Class 1 (Red) -> Price >= Open (Stock went up or stayed the same)
y = np.where(X[:, 1] < X[:, 0], 0, 1)

# Split the data into training and test sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Set up GridSearchCV to find the best k value
param_grid = {'n_neighbors': np.arange(1, 21)}
knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best k value and its accuracy
best_k = grid_search.best_params_['n_neighbors']
best_score = grid_search.best_score_

print(f"Best k value: {best_k}")
print(f"Best cross-validation accuracy: {best_score:.4f}")

# Train k-NN classifier with the best k value
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = best_knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", test_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot the accuracy scores for different k values
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(10, 6))
plt.plot(results['param_n_neighbors'], results['mean_test_score'], marker='o', linestyle='-')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validation Accuracy")
plt.title("k-NN Accuracy for Different k Values")
plt.grid(True)
plt.show()
