# Step 1: Import Required Libraries
import numpy as np
import pandas as pd

# Step 2: Upload the Excel File in Google Colab
from google.colab import files
uploaded = files.upload()

# Load the Excel file
file_name = list(uploaded.keys())[0]  # Get uploaded file name
xls = pd.ExcelFile(file_name)

# Step 3: Load the "IRCTC Stock Price" Sheet
df = xls.parse('IRCTC Stock Price')

# Step 4: Define Price Range Classes
def categorize_price(price):
    if price < 2000:
        return "Low"
    elif 2000 <= price < 2500:
        return "Mid"
    else:
        return "High"

df['Price Range'] = df['Price'].apply(categorize_price)


# Step 5: Select Features for Analysis and Preprocessing
features = ["Price", "Open", "High", "Low", "Volume", "Chg%"]

# Convert 'Volume' column to numeric by removing 'K' and 'M' and scaling
def convert_volume(volume_str):
    if isinstance(volume_str, str):   # Check if it's a string
        if 'K' in volume_str:
            return float(volume_str.replace('K', '')) * 1000
        elif 'M' in volume_str:
            return float(volume_str.replace('M', '')) * 1000000
        else:
            try:
                return float(volume_str)  # If no 'K' or 'M', try converting to float
            except ValueError:
                return np.nan  # If conversion fails, replace with NaN
    else:
        return volume_str   # If not a string, return original value

df['Volume'] = df['Volume'].apply(convert_volume)

# Step 6: Compute Mean & Standard Deviation for Each Class
class_groups = df.groupby("Price Range")[features]

class_centroids = class_groups.mean()   # Mean (Centroids)
class_spreads = class_groups.std()     # Standard Deviation (Spread)


# Step 7: Compute Euclidean Distance Between Class Centroids
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

classes = class_centroids.index.tolist()
interclass_distances = {}

for i in range(len(classes)):
    for j in range(i + 1, len(classes)):
        dist = euclidean_distance(class_centroids.loc[classes[i]], class_centroids.loc[classes[j]])
        interclass_distances[(classes[i], classes[j])] = dist

# Step 8: Print Results
print("\n### Class Centroids (Mean Values) ###")
print(class_centroids)

print("\n### Intraclass Spread (Standard Deviation) ###")
print(class_spreads)

print("\n### Interclass Distances ###")
for (cls1, cls2), distance in interclass_distances.items():
    print(f"Distance between {cls1} and {cls2}: {distance:.2f}")
