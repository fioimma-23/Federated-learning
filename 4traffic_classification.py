import pandas as pd

#Load
data = pd.read_csv('newcsv.csv')

#Display protocols
print("Unique Protocols in the dataset:", data['Protocol'].unique())

#label traffic
def label_traffic(row):
    if row['Protocol'] == 'TCP':
        return 'TCP Traffic'
    elif row['Protocol'] == 'UDP':
        return 'UDP Traffic'
    else:
        return 'Other'

#new column for labeled traffic
data['label'] = data.apply(label_traffic, axis=1)

# Save
data.to_csv('labeled_traffic_data.csv', index=False)

# Display classification
print("Traffic Classification Data:\n", data.head())
