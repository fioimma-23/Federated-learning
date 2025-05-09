import pandas as pd

# Load your dataset
data = pd.read_csv('newcsv.csv')  # Adjust filename as necessary

# View the first few rows
print("First few rows of the dataset:")
print(data.head())

# Check the data types
print("\nData types of each column:")
print(data.dtypes)

# List columns
print("\nColumns in the dataset:")
print(data.columns)

# Separate features and labels
X = data.drop('label', axis=1)  # Assuming 'label' is the label column
y = data['label']

print("\nFeatures (X):")
print(X.head())

print("\nLabels (y):")
print(y.head())

# Check unique labels
print("\nUnique labels in the dataset:")
print(y.unique())

# Get summary statistics for features
print("\nSummary statistics for features:")
print(X.describe())
