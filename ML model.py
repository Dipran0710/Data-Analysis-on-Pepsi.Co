#Dipran Bhandari S372252

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load your dataset
file_path = 'E:/ML/PEP.csv'
df = pd.read_csv(file_path)

# Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print("Missing values in each column:\n", df.isnull().sum())

# Example: Handle missing values (if any are present)
# Fill missing values with mean or median for numerical columns
df.fillna(df.mean(), inplace=True)

# If there are categorical variables, perform one-hot encoding
# Assuming 'Category' is a placeholder for any categorical column in your dataset
if 'Category' in df.columns:
    df = pd.get_dummies(df, columns=['Category'], drop_first=True)

# Check the dataset again after preprocessing
print(df.head())

# Assume 'Target' is the placeholder for the target variable in your dataset
# Modify based on your actual dataset structure

# Features and target variable selection
# Create a target variable for next day's closing price
df['Price_Change'] = df['Close'].diff().shift(-1)  # Price change for the next day
df['Target'] = (df['Price_Change'] > 0).astype(int)  # 1 if price goes up, 0 if it goes down
df.dropna(inplace=True)  # Remove NaN values after shift

# Features and target variable selection
X = df.drop(['Date', 'Target', 'Price_Change'], axis=1)  # Exclude 'Date' and target-related columns
y = df['Target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))