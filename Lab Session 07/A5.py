from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # Import StandardScaler
import matplotlib.pyplot as plt

# Assuming X contains your original data with categorical features

# 1. Create a StandardScaler object
scaler = StandardScaler()

# 2. Select only numerical features from X
X_numeric = X.select_dtypes(include=['number']) # Select numerical columns

# 3. Fit the scaler on numerical features and transform
X_scaled = scaler.fit_transform(X_numeric)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)  # Apply PCA to scaled numerical data

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)  # Use scaled data for clustering

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)  # Use scaled data for clustering

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
plt.title('K-Means Clustering')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='plasma', s=50, alpha=0.7)
plt.title('DBSCAN Clustering')
plt.show()

print("All tasks completed!")
