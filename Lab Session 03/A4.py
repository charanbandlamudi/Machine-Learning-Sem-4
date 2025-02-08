import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

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

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

score = knn.score(X_test, y_test)
print("Model R² Score:", score)
