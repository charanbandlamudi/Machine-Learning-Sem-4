import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

file_paths = ['255_s.xlsx', '259_s.xlsx', '261_s.xlsx', '285_s.xlsx', '287_s.xlsx',
              '219_student.xlsx', '220_student.xlsx', '221_student.xlsx', '222_student.xlsx', '223_student.xlsx']


df_list = [pd.read_excel(file, sheet_name='Sheet1', engine='openpyxl') for file in file_paths]
df = pd.concat(df_list, ignore_index=True)

# Encode categorical columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split features and target
features = df.drop('Number', axis=1)  # Replace 'Number' with your target column
target = df['Number']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Build Decision Tree
dt_model = DecisionTreeClassifier(criterion='entropy')
dt_model.fit(X_train, y_train)

# Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=features.columns, class_names=[str(cls) for cls in target.unique()], filled=True)
plt.show()