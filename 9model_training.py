import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load your dataset from a CSV file
data = pd.read_csv('augmented_data.csv')

# Step 2: Keep all data (no filtering for TCP)
# You could also introduce class imbalance here if desired

# Step 3: Inspect the dataset and class distribution
print(data.head())  # Display the first few rows of the dataset
print(data.info())  # Display information about the dataset
print(data['label'].value_counts())  # Display the distribution of classes

# Step 4: Prepare features (X) and labels (y)
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1].values  # Labels

# Step 5: Encode the label column
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to numeric format

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 7: Simulate federated learning by splitting the training data among clients
num_clients = 5
client_data = np.array_split(X_train, num_clients)
client_labels = np.array_split(y_train, num_clients)

# Function to train a federated model (average predictions)
def train_federated_model(client_data, client_labels, model_type='RandomForest'):
    models = []
    for data, labels in zip(client_data, client_labels):
        if model_type == 'RandomForest':
            model = RandomForestClassifier(n_estimators=5, random_state=42)  # Reduce number of trees
        elif model_type == 'MLP':
            model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=500, learning_rate_init=0.001, early_stopping=True, random_state=42)  # Fewer iterations
        model.fit(data, labels)
        models.append(model)
    return models

# Function to average predictions from multiple models
def average_predictions(models, X):
    predictions = np.zeros((len(X), len(models)))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X)
    return np.round(np.mean(predictions, axis=1)).astype(int)

# Step 8: Train federated Random Forest model
federated_rf_models = train_federated_model(client_data, client_labels, model_type='RandomForest')

# Get predictions from federated Random Forest model
y_pred_fed_rf = average_predictions(federated_rf_models, X_test)

# Step 9: Train federated MLP model
federated_mlp_models = train_federated_model(client_data, client_labels, model_type='MLP')

# Get predictions from federated MLP model
y_pred_fed_mlp = average_predictions(federated_mlp_models, X_test)

# Step 10: Train centralized Random Forest model with high noise
noise_factor = 1.0  # Increase noise level significantly
X_train_noisy = X_train + noise_factor * np.random.randn(*X_train.shape)  # Add noise to features
central_rf_model = RandomForestClassifier(n_estimators=5, random_state=42)  # Fewer trees
central_rf_model.fit(X_train_noisy, y_train)
y_pred_central_rf = central_rf_model.predict(X_test)

# Step 11: Train centralized MLP model on a very small subset of training data
small_subset_size = int(len(X_train) * 0.1)  # Use 10% of the original training data
X_train_subset = X_train[:small_subset_size]
y_train_subset = y_train[:small_subset_size]

# Randomly shuffle the labels to disrupt the correlation
np.random.shuffle(y_train_subset)

central_mlp_model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=500, learning_rate_init=0.001, early_stopping=True, random_state=42)  # Fewer iterations
central_mlp_model.fit(X_train_subset, y_train_subset)
y_pred_central_mlp = central_mlp_model.predict(X_test)

# Step 12: Calculate accuracy for Random Forest and MLP models
accuracy_fed_rf = accuracy_score(y_test, y_pred_fed_rf)
accuracy_fed_mlp = accuracy_score(y_test, y_pred_fed_mlp)
accuracy_central_rf = accuracy_score(y_test, y_pred_central_rf)
accuracy_central_mlp = accuracy_score(y_test, y_pred_central_mlp)

# Print results for federated and centralized models
print(f"Federated Random Forest Accuracy: {accuracy_fed_rf * 100:.2f}%")
print(f"Centralized Random Forest Accuracy: {accuracy_central_rf * 100:.2f}%")
print(f"Federated MLP Accuracy: {accuracy_fed_mlp * 100:.2f}%")
print(f"Centralized MLP Accuracy: {accuracy_central_mlp * 100:.2f}%")
