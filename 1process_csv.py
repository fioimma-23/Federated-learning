import pandas as pd

#read
df = pd.read_csv('newcsv.csv')

#display
print("First few rows of the dataset:")
print(df.head())

#statistics
print("\nSummary statistics for numeric columns:")
print(df.describe())

#number of occurrences of protocol
protocol_counts = df['Protocol'].value_counts()
print("\nNumber of occurrences for each protocol:")
print(protocol_counts)

df['Length'] = pd.to_numeric(df['Length'], errors='coerce')

#average packet length
avg_length_by_protocol = df.groupby('Protocol')['Length'].mean()
print("\nAverage packet length by protocol:")
print(avg_length_by_protocol)
