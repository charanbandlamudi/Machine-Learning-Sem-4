from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Compute metrics
silhouette = silhouette_score(X_train, kmeans.labels_)
ch_score = calinski_harabasz_score(X_train, kmeans.labels_)
db_index = davies_bouldin_score(X_train, kmeans.labels_)

# Output
print("\n Silhouette Score:", silhouette)
print("Calinski-Harabasz Score:", ch_score)
print(" Davies-Bouldin Index:", db_index)