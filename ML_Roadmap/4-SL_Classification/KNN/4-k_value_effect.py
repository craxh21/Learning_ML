import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic dataset (Fixed issue: removed n_classes)
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test different k values
k_values = range(1, 21)
train_accuracies = []
test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Compute accuracy
    train_accuracies.append(accuracy_score(y_train, knn.predict(X_train)))
    test_accuracies.append(accuracy_score(y_test, knn.predict(X_test)))

# Plot accuracy vs. k
plt.figure(figsize=(8, 5))
plt.plot(k_values, train_accuracies, label="Training Accuracy", marker='o')
plt.plot(k_values, test_accuracies, label="Testing Accuracy", marker='s')

plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("Choosing the Best k in KNN")
plt.legend()
plt.grid(True)
plt.show()
