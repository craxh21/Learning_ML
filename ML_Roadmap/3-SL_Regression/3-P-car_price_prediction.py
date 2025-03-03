# Polynomial Regression Project: Predict Car Price Depreciation

# Step 1: Import Libraries
import numpy as np
import pandas as pd
"""
✅Project – Predict Car Price Depreciation (Polynomial Regression)
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load or Create Dataset
# Sample synthetic dataset: Age of car (years) vs. Price ($1000s)
data = {
    'Age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Price': [45, 38, 32, 27, 23, 19, 16, 14, 12, 11]
}
df = pd.DataFrame(data)

# Step 3: Data Exploration
print("\nFirst few rows:")
print(df.head())

plt.scatter(df['Age'], df['Price'], color='blue')
plt.xlabel('Age of Car (years)')
plt.ylabel('Price ($1000s)')
plt.title('Car Price Depreciation')
plt.show()

# Step 4: Data Preprocessing
X = df[['Age']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build Polynomial Regression Model
degree = 3
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

# Step 6: Model Evaluation
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Degree {degree} Polynomial Regression:")
print(f"Train MSE: {train_mse:.4f}, Train R²: {train_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}, Test R²: {test_r2:.4f}")

# Step 7: Visualization
X_seq = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_seq_poly = poly_features.transform(X_seq)
y_seq_pred = model.predict(X_seq_poly)

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_seq, y_seq_pred, color='red', label=f'Polynomial Regression (degree={degree})')
plt.xlabel('Age of Car (years)')
plt.ylabel('Price ($1000s)')
plt.title('Polynomial Regression Fit')
plt.legend()
plt.show()
