import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# 1️⃣ Generate synthetic dataset
X, y = make_classification(n_samples=300, n_features=2, n_classes=2, n_redundant=0, random_state=42)

# 2️⃣ Train Decision Tree & Random Forest models
tree = DecisionTreeClassifier(random_state=42)
forest = RandomForestClassifier(n_estimators=50, random_state=42)

tree.fit(X, y)
forest.fit(X, y)

# 3️⃣ Plot decision boundaries
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

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_decision_boundary(tree, X, y, 'Decision Tree Decision Boundary')

plt.subplot(1, 2, 2)
plot_decision_boundary(forest, X, y, 'Random Forest Decision Boundary')

plt.show()


"""
Left Graph (Decision Tree):
The decision boundary has sharp edges and is more likely to overfit.
Right Graph (Random Forest):
The decision boundary is smoother, indicating better generalization.
The combination of multiple trees reduces variance and avoids overfitting.
"""