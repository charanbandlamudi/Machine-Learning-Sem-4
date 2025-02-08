import numpy as np
import pandas as pd

from google.colab import files
uploaded = files.upload()

file_name = list(uploaded.keys())[0]  
xls = pd.ExcelFile(file_name)

df = xls.parse('IRCTC Stock Price')

def categorize_price(price):
    if price < 2000:
        return "Low"
    elif 2000 <= price < 2500:
        return "Mid"
    else:
        return "High"

df['Price Range'] = df['Price'].apply(categorize_price)


features = ["Price", "Open", "High", "Low", "Volume", "Chg%"]

def convert_volume(volume_str):
    if isinstance(volume_str, str):  
        if 'K' in volume_str:
            return float(volume_str.replace('K', '')) * 1000
        elif 'M' in volume_str:
            return float(volume_str.replace('M', '')) * 1000000
        else:
            try:
                return float(volume_str)  
            except ValueError:
                return np.nan  
    else:
        return volume_str  

df['Volume'] = df['Volume'].apply(convert_volume)

class_groups = df.groupby("Price Range")[features]

class_centroids = class_groups.mean()  
class_spreads = class_groups.std()     


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

classes = class_centroids.index.tolist()
interclass_distances = {}

for i in range(len(classes)):
    for j in range(i + 1, len(classes)):
        dist = euclidean_distance(class_centroids.loc[classes[i]], class_centroids.loc[classes[j]])
        interclass_distances[(classes[i], classes[j])] = dist

print("\n### Class Centroids (Mean Values) ###")
print(class_centroids)

print("\n### Intraclass Spread (Standard Deviation) ###")
print(class_spreads)

print("\n### Interclass Distances ###")
for (cls1, cls2), distance in interclass_distances.items():
    print(f"Distance between {cls1} and {cls2}: {distance:.2f}")
