import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions

#gen synthetic data
x, y = make_classification(
    n_samples=300, n_features=2, n_classes=2, 
    n_redundant=0, n_clusters_per_class=1, random_state=42
)

#train decision tree model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(x, y)

plt.figure(figsize=(8,6))
plot_decision_regions(x, y, clf=model, legend=2)
plt.title("Decision Tree Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


"""
🔍 What This Code Does
1️⃣ Generates a binary classification dataset with 2 features.
2️⃣ Trains a Decision Tree model with max_depth=4.
3️⃣ Visualizes the decision boundary using plot_decision_regions.

✅ This shows how the tree splits the feature space into regions!
"""


"""
Decision Boundaries (Sharp Edges):
Unlike smooth boundaries in models like Logistic Regression or SVM with RBF kernels, Decision Trees create axis-aligned boundaries (vertical/horizontal splits).
This happens because the tree splits the feature space based on threshold conditions for each feature.


How the Splits Work:
The Decision Tree makes decisions in a hierarchical manner.
It first selects the most important feature and a threshold to split the data into two groups.
Then, it continues splitting each group further, creating box-like regions.
The max_depth=4 ensures that only 4 levels of splits are made, preventing overfitting.
"""