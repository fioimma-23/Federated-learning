# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the neural network model
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Custom dataset class
class AugmentedDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # Use LabelEncoder to convert string labels to integers
        self.label_encoder = LabelEncoder()
        self.data['label'] = self.label_encoder.fit_transform(self.data['label'])

        self.labels = torch.tensor(self.data['label'].values, dtype=torch.long)
        self.features = torch.tensor(self.data.drop(columns=['label']).values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Define the Client class
class Client:
    def __init__(self, id, model, data_loader):
        self.id = id
        self.model = model
        self.data_loader = data_loader

# Load the augmented dataset
csv_file = 'augmented_data.csv'
dataset = AugmentedDataset(csv_file)

# Setup parameters
num_clients = 5
input_size = dataset.features.shape[1]
num_classes = len(dataset.label_encoder.classes_)  # Get number of unique classes

# Ensure num_samples_per_client does not exceed dataset size
total_samples = len(dataset)
num_samples_per_client = min(200, total_samples // num_clients)

clients = []
for i in range(num_clients):
    # Split the dataset into client data
    client_data, _ = train_test_split(
        dataset.data, 
        test_size=total_samples - num_samples_per_client,  # Using total_samples to calculate test_size
        stratify=dataset.data['label'] if len(dataset.data['label'].unique()) > 1 else None, 
        random_state=i
    )
    
    # Reset index for client data
    client_data.reset_index(drop=True, inplace=True)
    
    # Create a DataLoader for the client
    client_model = SimpleNN(input_size, num_classes)
    client_loader = DataLoader(AugmentedDataset(csv_file=csv_file), batch_size=32, shuffle=True)  # Use the full dataset here
    clients.append(Client(id=i, model=client_model, data_loader=client_loader))

# Initialize a list to store results
results_list = []

# Training loop
for round_num in range(1, 6):
    print(f"Round {round_num}")
    for client in clients:
        client.model.train()
        for features, labels in client.data_loader:
            # Your training code here (forward pass, loss calculation, backpropagation, etc.)
            pass
        
        # Metrics calculation (this is a placeholder)
        predictions = client.model(client.data_loader.dataset.features)  # Example prediction line
        pred_labels = torch.argmax(predictions, dim=1)
        
        accuracy = accuracy_score(client.data_loader.dataset.labels.numpy(), pred_labels.numpy())
        precision = precision_score(client.data_loader.dataset.labels.numpy(), pred_labels.numpy(), average='weighted', zero_division=0)
        recall = recall_score(client.data_loader.dataset.labels.numpy(), pred_labels.numpy(), average='weighted', zero_division=0)
        f1 = f1_score(client.data_loader.dataset.labels.numpy(), pred_labels.numpy(), average='weighted', zero_division=0)
        
        # Print metrics
        print(f"Client {client.id}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Append the results to the list
        results_list.append({
            'Round': round_num,
            'Client': client.id,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })

# Convert results list to a DataFrame
results = pd.DataFrame(results_list)

# Save the results to a CSV file
results.to_csv('federated_learning_results.csv', index=False)

