import pandas as pd

# Load the dataset
data_path = 'data/training.1600000.processed.noemoticon.csv'
df = pd.read_csv(data_path, encoding='latin1')  # Use 'latin1' to avoid encoding issues

# Display the first few rows
print(df.head())

# Check for column names and null values
print("Columns:", df.columns)
print("\nMissing values:\n", df.isnull().sum())

