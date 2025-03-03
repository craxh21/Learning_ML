#using t-SNE to visualize handwritten digits (MNIST dataset) in 2D.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load the MNIST dataset (Handwritten digits)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# 2️⃣ Take a smaller subset (10,000 samples for speed)
X_subset = X[:1000]
y_subset = y[:1000]

# 3️⃣ Normalize the data (t-SNE performs better on scaled data)
X_scaled = StandardScaler().fit_transform(X_subset)

# 4️⃣ Apply t-SNE to reduce from 784D to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# 5️⃣ Plot the t-SNE visualization
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label="Digit Labels")
plt.title("t-SNE Visualization of MNIST Digits")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()

"""
🔹 Each point represents a digit (0-9) in reduced 2D space.
🔹 Clusters indicate that similar digits are grouped together.

 Well-Separated vs. Close Clusters
Digits like 0 & 1 might be far apart because they look visually distinct.
Digits like 5 & 6 might be closer because they share similar curves.
"""

"""
Axes Don’t Have a Fixed Meaning

Unlike PCA, the x and y axes don’t directly represent features.
They only help visualize relationships between points, meaning distances matter more than absolute positions.
"""