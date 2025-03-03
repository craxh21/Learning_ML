import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Actual vs. Predicted values
y_true = np.array([3, 5, 7, 9])
y_pred = np.array([2.8, 5.5, 6.8, 9.2])

# Calculate metrics
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
