import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate synthetic data (points along a diagonal line)
np.random.seed(42)
X = np.random.multivariate_normal(mean=[5, 5], cov=[[3, 2], [2, 3]], size=100)

# Apply PCA
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# Get principal components
pc1 = pca.components_[0]  # First principal component
pc2 = pca.components_[1]  # Second principal component
mean = pca.mean_  # Mean of the data

# Plot original data
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, label="Original Data")

# Plot principal components
plt.arrow(mean[0], mean[1], pc1[0]*3, pc1[1]*3, color='red', width=0.1, label="PC1 (Most Important)")
plt.arrow(mean[0], mean[1], pc2[0]*3, pc2[1]*3, color='blue', width=0.05, label="PC2 (Less Important)")

# Project data onto the first principal component
X_projected = np.outer(X_pca[:, 0], pc1) + mean
plt.scatter(X_projected[:, 0], X_projected[:, 1], color='green', alpha=0.6, label="Projected Data (1D)")

plt.legend()
plt.title("PCA Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid()
plt.show()

"""
The original data (blue points) is spread across 2D space.
The red arrow (PC1) shows the most important direction (maximum variance).
The blue arrow (PC2) is the second direction (less important).
The green points represent the data after reducing it to 1D (projected onto PC1).
"""