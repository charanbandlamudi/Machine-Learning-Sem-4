import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from google.colab import files
uploaded = files.upload()

file_name = list(uploaded.keys())[0]
df = pd.ExcelFile(file_name)

df_irctc = pd.read_excel(file_name, sheet_name='IRCTC Stock Price')

df_irctc = df_irctc[['Open', 'Price']].dropna()

df_irctc['Open'] = pd.to_numeric(df_irctc['Open'], errors='coerce')
df_irctc['Price'] = pd.to_numeric(df_irctc['Price'], errors='coerce')

df_irctc = df_irctc.dropna()

X_train = df_irctc[['Open', 'Price']].values[:20]

y_train = np.where(X_train[:, 1] < X_train[:, 0], 0, 1)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

x_test_values = np.arange(0, 10.1, 0.1)
y_test_values = np.arange(0, 10.1, 0.1)
X_test, Y_test = np.meshgrid(x_test_values, y_test_values)
X_test_flat = X_test.ravel()
Y_test_flat = Y_test.ravel()
test_points = np.c_[X_test_flat, Y_test_flat]

predicted_classes = knn.predict(test_points)
predicted_classes = predicted_classes.reshape(X_test.shape)

plt.figure(figsize=(10, 8))
plt.contourf(X_test, Y_test, predicted_classes, alpha=0.5, cmap='bwr')
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Class 0 (Train - Blue)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Class 1 (Train - Red)')

plt.title("k-NN Classification (k=3) - IRCTC Stock Price")
plt.xlabel("Open Price")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
plt.show()
