import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# Generate non-linearly separable data
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Create SVM models with different kernels
kernels = ['linear', 'poly', 'rbf']
titles = ['Linear Kernel', 'Polynomial Kernel (degree=3)', 'RBF Kernel']

plt.figure(figsize=(15, 5))

for i, kernel in enumerate(kernels):
    model = SVC(kernel=kernel, degree=3)  # degree is ignored for 'linear' and 'rbf'
    model.fit(X, y)

    # Create mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Predict values for the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.subplot(1, 3, i + 1)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title(titles[i])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
