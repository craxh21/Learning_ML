import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data
data = pd.DataFrame({
    'Hours_Studied': [1, 2, 3, 4, 5],
    'Classes_Attended': [3, 4, 2, 5, 4],
    'Exam_Score': [50, 60, 55, 70, 65]
})

X = data[['Hours_Studied', 'Classes_Attended']]
y = data['Exam_Score']

# Train model
model = LinearRegression()
model.fit(X, y)

# Coefficients
print("Coefficients:", model.coef_)#Coefficients: Impact of each feature on predictions.
print("Intercept:", model.intercept_)#Intercept: Base score when features are zero.
#Intercept (c): Prediction when all features are zero.
#Coefficients (m): Change in prediction per unit increase in the feature.