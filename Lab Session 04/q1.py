import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from google.colab import files
uploaded = files.upload()

file_name = list(uploaded.keys())[0]
excel_data = pd.ExcelFile(file_name)

df = pd.read_excel(file_name, sheet_name='IRCTC Stock Price')

df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna()  
df['Next_Day_Price'] = df['Price'].shift(-1) 
df['Target'] = (df['Next_Day_Price'] > df['Price']).astype(int)  
df = df.dropna()  

X = df[['Open', 'High', 'Low', 'Price']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

conf_matrix_train = confusion_matrix(y_train, y_train_pred)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

report_train = classification_report(y_train, y_train_pred)
report_test = classification_report(y_test, y_test_pred)

print("Confusion Matrix - Training Data:")
print(conf_matrix_train)
print("\nClassification Report - Training Data:")
print(report_train)

print("Confusion Matrix - Test Data:")
print(conf_matrix_test)
print("\nClassification Report - Test Data:")
print(report_test)
