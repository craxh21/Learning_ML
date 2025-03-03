import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

# 1️⃣ Generate synthetic data
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.0, random_state=42)

# 2️⃣ Compute linkage matrix for dendrogram
linkage_matrix = linkage(X, method='ward')

# 3️⃣ Plot Dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# 4️⃣ Apply Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=4, linkage='ward')
clusters = model.fit_predict(X)

# 5️⃣ Scatter plot of clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=clusters, palette="viridis", s=70)
plt.title("Clusters Identified by Hierarchical Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


"""
Explanation of Output
1️⃣ Dendrogram: Shows how clusters are merged at each step. The higher the merge, the more different the clusters.
2️⃣ Scatter Plot: Displays the final clusters after hierarchical clustering.
"""