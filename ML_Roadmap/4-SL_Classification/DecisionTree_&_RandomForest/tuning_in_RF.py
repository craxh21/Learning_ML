import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# 1️⃣ Generate synthetic dataset
X, y = make_classification(n_samples=300, n_features=2, n_classes=2, n_redundant=0, random_state=42)

# 2️⃣ Train Random Forest models with different hyperparameters
forest1 = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
forest2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

forest1.fit(X, y)
forest2.fit(X, y)

# 3️⃣ Function to plot decision boundary
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

# 4️⃣ Plot decision boundaries for different hyperparameters
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_decision_boundary(forest1, X, y, 'RF: n_estimators=10, max_depth=2')

plt.subplot(1, 2, 2)
plot_decision_boundary(forest2, X, y, 'RF: n_estimators=100, max_depth=10')

plt.show()

"""
Left Graph (n_estimators=10, max_depth=2):
Simpler decision boundary with smoother transitions.
Less likely to overfit but may not capture complex patterns.

Right Graph (n_estimators=100, max_depth=10):
More detailed boundary with better adaptation to data.
Higher risk of overfitting if max_depth is too large.

"""