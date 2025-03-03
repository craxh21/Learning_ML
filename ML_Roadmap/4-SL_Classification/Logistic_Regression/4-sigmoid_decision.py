import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# 1️⃣ Plot the Sigmoid Function
z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(z, sigmoid, color='blue')
plt.axvline(0, color='red', linestyle='--', label='Decision Threshold (z=0)')
plt.axhline(0.5, color='green', linestyle='--')
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('Sigmoid(z)')
plt.legend()
plt.grid(True)

# 2️⃣ Visualizing Decision Boundary on a Dataset
# Create synthetic dataset
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Plot data points and decision boundary
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', label='Data Points')

# Create mesh grid for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Predict probabilities for grid points
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Plot decision boundary (where probability=0.5)
contour = plt.contour(xx, yy, Z, levels=[0.5], cmap="Greys", linestyles='dashed')
plt.clabel(contour, fmt={'0.5': 'Decision Boundary'})

plt.title('Decision Boundary in Logistic Regression')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.tight_layout()
plt.show()
