import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function (Log Loss)
def compute_cost(X, y, weights):
    m = len(y)
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    
    # Log loss calculation
    cost = -(1/m) * np.sum(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
    return cost

# Gradient descent to update weights
def gradient_descent(X, y, weights, learning_rate, epochs):
    m = len(y)
    cost_history = []

    for i in range(epochs):
        z = np.dot(X, weights)
        predictions = sigmoid(z)

        # Gradient calculation
        gradient = (1/m) * np.dot(X.T, (predictions - y))
        
        # Update weights
        weights -= learning_rate * gradient

        # Compute and store the cost for debugging
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

        # Debug logs every 100 iterations
        if i % 100 == 0 or i == epochs - 1:
            print(f"Epoch {i+1}: Cost = {cost:.4f}")
    
    return weights, cost_history

# Generate synthetic data
np.random.seed(42)
num_samples = 200
X = np.random.randn(num_samples, 2)
true_weights = np.array([2, -1])
true_bias = -0.5

# Generate labels based on linear combination + noise
z = np.dot(X, true_weights) + true_bias
y = (sigmoid(z) >= 0.5).astype(int)

# Add intercept term to X
X_with_bias = np.hstack([np.ones((num_samples, 1)), X])

# Initialize weights (including bias term)
initial_weights = np.zeros(X_with_bias.shape[1])
learning_rate = 0.1
epochs = 1000

# Train the model using gradient descent
final_weights, cost_history = gradient_descent(X_with_bias, y, initial_weights, learning_rate, epochs)

# Plot cost reduction over epochs
plt.plot(range(epochs), cost_history, color='blue')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.grid(True)
plt.show()

# Final weights and bias
print(f"Final weights: {final_weights[1:]}")
print(f"Final bias: {final_weights[0]}")
