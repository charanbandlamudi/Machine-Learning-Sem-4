import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from google.colab import files
uploaded = files.upload()

file_name = list(uploaded.keys())[0]
df = pd.ExcelFile(file_name)

df_irctc = pd.read_excel(file_name, sheet_name='IRCTC Stock Price')

df_irctc = df_irctc[['Open', 'Price']].dropna()

df_irctc['Open'] = pd.to_numeric(df_irctc['Open'], errors='coerce')
df_irctc['Price'] = pd.to_numeric(df_irctc['Price'], errors='coerce')

df_irctc = df_irctc.dropna()

X = df_irctc[['Open', 'Price']].values[:100]

y = np.where(X[:, 1] < X[:, 0], 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {'n_neighbors': np.arange(1, 21)}
knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_['n_neighbors']
best_score = grid_search.best_score_

print(f"Best k value: {best_k}")
print(f"Best cross-validation accuracy: {best_score:.4f}")

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

y_pred = best_knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", test_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(10, 6))
plt.plot(results['param_n_neighbors'], results['mean_test_score'], marker='o', linestyle='-')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validation Accuracy")
plt.title("k-NN Accuracy for Different k Values")
plt.grid(True)
plt.show()
