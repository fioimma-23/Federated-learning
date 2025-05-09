import pandas as pd

# Load
data = pd.read_csv('labeled_traffic_data.csv')

#Time to float
data['Time'] = data['Time'].astype(float)

# flow duration and inter-arrival time
data['flow_duration'] = data['Time'].diff().fillna(0)
data['inter_arrival_time'] = data['Time'].diff().fillna(0)

# Length
data['Length'] = (data['Length'] - data['Length'].mean()) / data['Length'].std()

# Label traffic
data['label'] = data['Protocol'].apply(lambda x: 'TCP Traffic' if 'TCP' in x else ('UDP Traffic' if 'UDP' in x else 'Other'))

# Filter
data = data[data['label'].isin(['TCP Traffic', 'UDP Traffic'])]

# final features
final_features = data[['Length', 'flow_duration', 'inter_arrival_time', 'label']]

# Save
final_features.to_csv('processed_data.csv', index=False)

# Display
print("Feature Extraction Results:")
print(final_features.head(10))

# Summary statistics
print("\nSummary Statistics:")
print(final_features.describe())

# Unique labels
print("\nUnique labels in the dataset:")
print(final_features['label'].unique())
