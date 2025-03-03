#20-2-25
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Sample dataset with missing values
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, None, 30, None, 28],
    'Salary': [50000, 60000, None, 58000, 62000]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)


# Fill missing values with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].median(), inplace=True)
print("\nData after filling missing values:\n", df)


# Add a categorical column
df['Department'] = ['HR', 'Engineering', 'HR', 'Marketing', 'Engineering']
# One-hot encode 'Department'
df_encoded = pd.get_dummies(df, columns=['Department'])
print("\nData after one-hot encoding:\n", df_encoded)


#feature scaling
scaler = StandardScaler()
df_encoded[['Age', 'Salary']] = scaler.fit_transform(df_encoded[['Age', 'Salary']])

print("\nData after feature scaling:\n", df_encoded)


# Create a new feature: Salary per Age
df_encoded['Salary_per_Age'] = df_encoded['Salary'] / (df_encoded['Age'] + 1)  # +1 to avoid division by zero
print("\nData after feature engineering:\n", df_encoded)


# Visualize outliers using boxplot
sns.boxplot(x=df_encoded['Salary'])
plt.title("Salary Outlier Detection")
plt.show()

# Remove outliers beyond 1.5*IQR
Q1 = df_encoded['Salary'].quantile(0.25)
Q3 = df_encoded['Salary'].quantile(0.75)
IQR = Q3 - Q1

df_no_outliers = df_encoded[(df_encoded['Salary'] >= Q1 - 1.5 * IQR) & (df_encoded['Salary'] <= Q3 + 1.5 * IQR)]
print("\nData after removing outliers:\n", df_no_outliers)


# Features (X) and target (y)
X = df_no_outliers.drop(columns=['Salary'])
y = df_no_outliers['Salary']

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)} | Test set size: {len(X_test)}")


# Basic statistics of numerical features
print(df_no_outliers.describe())


# Histogram for Age distribution
sns.histplot(df_no_outliers['Age'], bins=5, kde=True)
plt.title("Age Distribution")
plt.show()


# Scatter plot: Age vs. Salary
sns.scatterplot(x=df_no_outliers['Age'], y=df_no_outliers['Salary'])
plt.title("Age vs. Salary")
plt.show()

# Correlation heatmap
corr = df_no_outliers.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()


# Bar plot for Department distribution
sns.countplot(x=df['Department'])
plt.title("Department Distribution")
plt.show()
