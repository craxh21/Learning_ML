import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])  # Hours studied
y = np.array([2, 4, 5, 4, 5])            # Exam scores

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Calculate MSE:  Calculates the average squared difference between actual and predicted values.
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error (MSE):", mse)

# Plot
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.title('Simple Linear Regression')
plt.show()
