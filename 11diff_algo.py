import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the augmented dataset
data = pd.read_csv('augmented_data.csv')
X = data[['Length', 'flow_duration', 'inter_arrival_time']].values
y = data['label'].values

# Convert labels to binary (e.g., 0 for TCP Traffic, 1 for TLS Traffic)
y = np.where(y == 'TCP Traffic', 0, 1)  # Adjust based on your label encoding

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)  # Binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)  # Reshape for binary output
        outputs = model(X_test_tensor)
        predictions = (outputs >= 0.5).float()  # Binarize predictions
        accuracy = accuracy_score(y_test_tensor.numpy(), predictions.numpy())
    return accuracy

# Function to simulate federated training
def federated_training(num_clients=5, rounds=10, epochs=5, learning_rate=0.01, method='FedAvg', mu=0.1):
    # Convert data to PyTorch tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).view(-1, 1)  # Reshape for binary output

    # Split data among clients
    client_data = np.array_split(X_tensor.numpy(), num_clients)
    client_labels = np.array_split(y_tensor.numpy(), num_clients)

    # Initialize global model
    global_model = SimpleNN(input_size=X_train.shape[1])
    criterion = nn.BCELoss()
    
    # Store accuracies for each round
    accuracies = []

    # Federated training rounds
    for round_num in range(rounds):
        print(f'Round {round_num + 1}/{rounds}')
        local_models = []
        
        # Simulate training on each client
        for client in range(num_clients):
            local_model = SimpleNN(input_size=X_train.shape[1])
            optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
            X_client = torch.FloatTensor(client_data[client])
            y_client = torch.FloatTensor(client_labels[client])

            # Train local model
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = local_model(X_client)
                loss = criterion(outputs, y_client)
                loss.backward()
                optimizer.step()
            
            local_models.append(local_model.state_dict())

        # Federated averaging or proximal aggregation
        global_state_dict = global_model.state_dict()
        for key in global_state_dict.keys():
            if method == 'FedAvg':
                global_state_dict[key] = torch.mean(torch.stack(
                    [local_models[client][key] for client in range(num_clients)]), dim=0)
            elif method == 'FedProx':
                proximal_weights = torch.mean(torch.stack(
                    [local_models[client][key] for client in range(num_clients)]), dim=0)
                global_state_dict[key] = proximal_weights - mu * (proximal_weights - global_state_dict[key])
        
        global_model.load_state_dict(global_state_dict)

        # Evaluate the global model after each round
        accuracy = evaluate_model(global_model, X_test, y_test)
        accuracies.append(accuracy)
        print(f'Accuracy: {accuracy * 100:.2f}%')  # Print accuracy as percentage

    return global_model, accuracies

# Train the global model using FedAvg
print("Training with FedAvg")
trained_model_fedavg, accuracies_fedavg = federated_training(num_clients=5, rounds=10, epochs=5, method='FedAvg')

# Print FedAvg mean accuracy
mean_accuracy_fedavg = np.mean(accuracies_fedavg) * 100  # Calculate mean accuracy
print(f"\nFedAvg Accuracies")
print(f'Mean Accuracy: {mean_accuracy_fedavg:.2f}%')  # Print mean accuracy

# Train the global model using FedProx
print("\nTraining with FedProx")
trained_model_fedprox, accuracies_fedprox = federated_training(num_clients=5, rounds=10, epochs=5, method='FedProx')

# Print FedProx mean accuracy
mean_accuracy_fedprox = np.mean(accuracies_fedprox) * 100  # Calculate mean accuracy
print(f"\nFedProx Accuracies")
print(f'Mean Accuracy: {mean_accuracy_fedprox:.2f}%')  # Print mean accuracy
