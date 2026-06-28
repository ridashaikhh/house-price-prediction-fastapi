from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load the California Housing dataset
data = fetch_california_housing()

# Create a DataFrame using the feature names
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add the target variable (house prices)
df["Price"] = data.target

# Display dataset information
print("Shape of Dataset:", df.shape)

print("\nFirst Five Rows")
print(df.head())

print("\nLast Five Rows")
print(df.tail())

print("\nColumn Names")
print(df.columns)

print("\nData Types")
print(df.dtypes)

print("\nChecking Missing Values")
print(df.isnull().sum())

print("\nChecking Duplicate Rows")
print(df.duplicated().sum())

print("\nStatistical Summary")
print(df.describe())

print("\nDataset Information")
print(df.info())