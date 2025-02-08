import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from google.colab import files
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
xls = pd.ExcelFile(file_name)

df = pd.read_excel(xls, sheet_name="IRCTC Stock Price")

X = df[['Open', 'Low']]
y = df['High']

X = X.select_dtypes(include=[np.number])
y = pd.to_numeric(y, errors='coerce')

df = df.dropna(subset=['Open', 'Low', 'High'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_values = range(1, 12)
scores = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    scores.append(score)
    print(f"k={k}, R² Score: {score}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, scores, marker='o', linestyle='dashed', color='b')
plt.xlabel('k Value')
plt.ylabel('R² Score')
plt.title('kNN Accuracy for Different k Values')
plt.grid(True)
plt.show()

