import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Load the augmented dataset
data = pd.read_csv('augmented_data.csv')

# Step 2: Preprocess the data
# Encoding categorical labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Splitting features and labels
X = data.drop('label', axis=1)
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Train a simple neural network model
model = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', max_iter=200, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the model on the test set
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 5: Save the trained model
joblib.dump(model, 'traffic_classification_model.pkl')
print("Model saved as 'traffic_classification_model.pkl'.")

# Step 6: Load the saved model
loaded_model = joblib.load('traffic_classification_model.pkl')
print("Model loaded from 'traffic_classification_model.pkl'.")

# Step 7: Make predictions with the loaded model
loaded_y_pred = loaded_model.predict(X_test)

# Evaluate the loaded model
loaded_test_accuracy = accuracy_score(y_test, loaded_y_pred)
print(f"Loaded Model Test Accuracy: {loaded_test_accuracy * 100:.2f}%")
print("Loaded Model Classification Report:")
print(classification_report(y_test, loaded_y_pred, target_names=label_encoder.classes_))
