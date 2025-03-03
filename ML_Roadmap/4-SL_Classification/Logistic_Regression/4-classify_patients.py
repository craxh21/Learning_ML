# 🚀 **Classify Patients as Diabetic or Not - Complete Code**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc

# 1️⃣ Load & Explore the Dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(url, header=None, names=columns)

print("\nFirst 5 rows:")
print(data.head())
print("\nDataset Info:")
print(data.info())
print("\nMissing Values:")
print((data == 0).sum())  # Some zeros represent missing values

# 2️⃣ Data Preprocessing
# Replace zeros with median for certain columns
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zeros:
    data[col] = data[col].replace(0, data[col].median())

# Feature scaling
scaler = StandardScaler()
X = data.drop("Outcome", axis=1)
X_scaled = scaler.fit_transform(X)
y = data["Outcome"]

# 3️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4️⃣ Model Training (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

print("\nModel Coefficients:")
for feature, coef in zip(columns[:-1], model.coef_[0]):
    print(f"{feature}: {coef:.4f}")

# 5️⃣ Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 6️⃣ Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

"""
✅ **This code covers:**  
- Loading and cleaning data  
- Feature scaling and train-test split  
- Training Logistic Regression  
- Evaluating with Accuracy, Precision, Recall, F1-Score  
- Visualizing Confusion Matrix & ROC Curve  

Let me know if you need **explanations** or **further enhancements**!
"""


"""
1️⃣ Confusion Matrix (Left Plot)
What it shows:
The number of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) in a matrix form.

Interpretation:

TP (Top-left cell): Correctly predicted diabetic patients.
TN (Bottom-right cell): Correctly predicted non-diabetic patients.
FP (Top-right cell): Non-diabetic patients incorrectly classified as diabetic.
FN (Bottom-left cell): Diabetic patients incorrectly classified as non-diabetic.
Goal:
High TP and TN values are desirable, while FP and FN should be minimized.

2️⃣ ROC Curve (Right Plot)
What it shows:
The trade-off between True Positive Rate (TPR) and False Positive Rate (FPR) at various threshold settings.

Key Points:

The red diagonal line represents random guessing (AUC = 0.5).
The blue curve represents the model’s performance.
Higher curves (closer to the top-left corner) indicate better performance.
AUC (Area Under the Curve):

Ranges from 0 to 1.
AUC > 0.8: Good model
AUC = 0.5: No discrimination (random)
"""