import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# Generate synthetic classification data
X, y = make_classification(n_samples=200, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)

# Define different values of gamma for comparison
gamma_values = [0.1, 1, 10]

plt.figure(figsize=(15, 5))

for i, gamma in enumerate(gamma_values):
    model = SVC(kernel='rbf', gamma=gamma)
    model.fit(X, y)

    # Plot decision boundary
    plt.subplot(1, 3, i + 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Predict over mesh grid
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    plt.title(f'SVM with RBF Kernel (Gamma={gamma})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

"""
 Expected Output:
Gamma = 0.1 (Underfitting) → The decision boundary is too smooth, leading to errors.
Gamma = 1 (Balanced) → A reasonable boundary.
Gamma = 10 (Overfitting) → The boundary is too curvy, capturing noise.
"""