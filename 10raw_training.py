import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Step 1: Generate random data
num_samples = 1000  # Number of data points
num_features = 20   # Number of features

# Random feature data (X) and binary labels (y)
X_random = np.random.rand(num_samples, num_features)  # Random float values between 0 and 1
y_random = np.random.randint(0, 2, num_samples)       # Random binary labels (0 or 1)

# Step 2: Split the random data into training and testing sets
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(X_random, y_random, test_size=0.2, random_state=42)

# Step 3: Simulate federated learning by splitting the training data among clients
num_clients = 5
client_data = np.array_split(X_train_random, num_clients)
client_labels = np.array_split(y_train_random, num_clients)

# Function to train a federated model (average predictions)
def train_federated_model(client_data, client_labels, model_type='RandomForest'):
    models = []
    for data, labels in zip(client_data, client_labels):
        if model_type == 'RandomForest':
            model = RandomForestClassifier(n_estimators=10, random_state=42)
        elif model_type == 'MLP':
            model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, learning_rate_init=0.001, early_stopping=True, random_state=42)
        model.fit(data, labels)
        models.append(model)
    return models

# Function to average predictions from multiple models
def average_predictions(models, X):
    predictions = np.zeros((len(X), len(models)))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X)
    return np.round(np.mean(predictions, axis=1)).astype(int)

# Step 4: Train federated Random Forest model
federated_rf_models = train_federated_model(client_data, client_labels, model_type='RandomForest')

# Get predictions from federated Random Forest model
y_pred_fed_rf = average_predictions(federated_rf_models, X_test_random)

# Step 5: Train federated MLP model
federated_mlp_models = train_federated_model(client_data, client_labels, model_type='MLP')

# Get predictions from federated MLP model
y_pred_fed_mlp = average_predictions(federated_mlp_models, X_test_random)

# Step 6: Train centralized Random Forest model
central_rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
central_rf_model.fit(X_train_random, y_train_random)
y_pred_central_rf = central_rf_model.predict(X_test_random)

# Step 7: Train centralized MLP model
central_mlp_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, learning_rate_init=0.001, early_stopping=True, random_state=42)
central_mlp_model.fit(X_train_random, y_train_random)
y_pred_central_mlp = central_mlp_model.predict(X_test_random)

# Step 8: Calculate accuracy for Random Forest and MLP models
accuracy_fed_rf = accuracy_score(y_test_random, y_pred_fed_rf)
accuracy_fed_mlp = accuracy_score(y_test_random, y_pred_fed_mlp)
accuracy_central_rf = accuracy_score(y_test_random, y_pred_central_rf)
accuracy_central_mlp = accuracy_score(y_test_random, y_pred_central_mlp)

# Print results for federated and centralized models
print(f"Federated Random Forest Accuracy: {accuracy_fed_rf * 100:.2f}%")
print(f"Centralized Random Forest Accuracy: {accuracy_central_rf * 100:.2f}%")
print(f"Federated MLP Accuracy: {accuracy_fed_mlp * 100:.2f}%")
print(f"Centralized MLP Accuracy: {accuracy_central_mlp * 100:.2f}%")

