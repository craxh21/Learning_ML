import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Parameters
m, c = 0, 0            # Initial slope & intercept
alpha = 0.01           # Learning rate
epochs = 1000          # Number of iterations
n = len(X)             # Number of data points

# Gradient Descent
for _ in range(epochs):
    y_pred = m * X + c
    dm = (-2/n) * sum(X * (y - y_pred))  # Derivative w.r.t m
    dc = (-2/n) * sum(y - y_pred)        # Derivative w.r.t c
    m -= alpha * dm                      # Update m
    c -= alpha * dc                      # Update c

print(f"Slope (m): {m:.2f}, Intercept (c): {c:.2f}")

# Plotting
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, m*X + c, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Gradient Descent - Linear Regression')
plt.show()

"""
✅ Explanation:

Updates happen iteratively until MSE is minimized.
Red line → Final best-fit line after gradient descent.
"""