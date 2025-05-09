import pandas as pd
from sklearn.model_selection import train_test_split

#Load
combined_data = pd.read_csv('augmented_data.csv')

#Separate features and labels
X = combined_data[['Length', 'flow_duration', 'inter_arrival_time']]
y = combined_data['label']

#Split the data training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
