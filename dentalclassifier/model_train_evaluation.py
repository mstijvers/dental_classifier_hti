import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from tanden import SimpleCNN
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import cv2
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

# Define class labels
class_labels = ["gingivitis", "hypodontia", "discoloration", "caries", "calculus", "healthyteeth"]

# Create empty lists to store image and label pairs
data = []
labels = []

# Relative path to the data folder
data_folder = "Dentaldata"

# Modify the code for loading images
for class_label in class_labels:
    class_folder = os.path.join(data_folder, class_label)
    for filename in os.listdir(class_folder):
        if filename.endswith(".jpg"):  # Assuming image files end with .jpg
            img_path = os.path.join(class_folder, filename)
            try:
                with Image.open(img_path) as img:  # Use the with statement to load the image
                    if img is not None and len(np.array(img).shape) == 3:
                        data.append(img)
                        labels.append(class_labels.index(class_label))  # Use the index of the class as the label
                        print(f"Loaded image: {img_path}, Label: {class_labels.index(class_label)}")
                    else:
                        print(f"Skipped invalid image: {img_path}")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

# Convert data to NumPy arrays
data = np.array(data, dtype=object)
labels = np.array(labels)

# Add the following code to check the data
print(f"Total images: {len(data)}")
print(f"Total labels: {len(labels)}")
for class_label in class_labels:
    class_count = (labels == class_labels.index(class_label)).sum()
    print(f"Total images in class '{class_label}': {class_count}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create the model
num_classes = 6  # Assuming there are 6 classes
model = SimpleCNN(num_classes)

# Modify the transformation to handle RGBA images by converting them to RGB
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :]),  # Remove the alpha channel (retain RGB channels only)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Convert X_train images to PIL Images before applying the transformations
X_train_pil = [Image.fromarray(np.array(image)) for image in X_train]
X_train_tensor = torch.stack([transform(image) for image in X_train_pil])

# Convert X_test images to PIL Images before applying the transformations
X_test_pil = [Image.fromarray(np.array(image)) for image in X_test]
X_test_tensor = torch.stack([transform(image) for image in X_test_pil])

# Create data loaders for both the training and testing datasets
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, torch.tensor(y_train))
test_dataset = TensorDataset(X_test_tensor, torch.tensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20  # Choose the number of training epochs as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print the loss for each epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Save the model's weights and structure
torch.save(model.state_dict(), 'with_healthyteeth_model_20epochs.pth')

# Save the class label mapping for use during inference
import json
with open('class_labels.json', 'w') as f:
    json.dump(class_labels, f)

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0

predictions = []
true_labels = []


with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(true_labels, predictions)


accuracy = accuracy_score(true_labels, predictions)
error_rate = 1 - accuracy
recall = recall_score(true_labels, predictions, average='weighted')
precision = precision_score(true_labels, predictions, average='weighted')


criterion = nn.CrossEntropyLoss() 
total_loss = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

average_loss = total_loss / len(test_loader)


print("Confusion Matrix:")
print(cm)
print(f"Error Rate: {error_rate * 100:.2f}%")
print(f"Model Loss: {average_loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")






