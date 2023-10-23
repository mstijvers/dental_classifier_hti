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
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from torch.optim.lr_scheduler import StepLR  # 导入 StepLR 学习率调度器

# Define class labels
class_labels = ["gingivitis", "hypodontia", "discoloration", "caries", "calculus", "healthyteeth"]

# Create empty lists to store image and label pairs
data = []
labels = []

# Relative path to the data folder
data_folder = "Dentaldata"

# Load and preprocess images
for class_label in class_labels:
    class_folder = os.path.join(data_folder, class_label)
    for filename in os.listdir(class_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(class_folder, filename)
            try:
                with Image.open(img_path) as img:
                    if img is not None and len(np.array(img).shape) == 3:
                        data.append(img)
                        labels.append(class_labels.index(class_label))
                        print(f"Loaded image: {img_path}, Label: {class_labels.index(class_label)}")
                    else:
                        print(f"Skipped invalid image: {img_path}")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

# Convert data to NumPy arrays
data = np.array(data, dtype=object)
labels = np.array(labels)

# Print dataset statistics
print(f"Total images: {len(data)}")
print(f"Total labels: {len(labels)}")
for class_label in class_labels:
    class_count = (labels == class_labels.index(class_label)).sum()
    print(f"Total images in class '{class_label}': {class_count}")

# Split the dataset into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create the model
num_classes = 6
model = SimpleCNN(num_classes)

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create data loaders for training, validation, and testing datasets
batch_size = 32
train_dataset = TensorDataset(torch.stack([transform(img) for img in X_train]), torch.tensor(y_train))
val_dataset = TensorDataset(torch.stack([transform(img) for img in X_val]), torch.tensor(y_val))
test_dataset = TensorDataset(torch.stack([transform(img) for img in X_test]), torch.tensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create StepLR learning rate scheduler
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

best_val_loss = float('inf')  # Track the best validation loss

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Adjust learning rate
    scheduler.step()

    # Print training loss and learning rate
    print(f"Epoch {epoch + 1}/{num_epochs}, LR: {scheduler.get_last_lr()[0]}, Loss: {running_loss / len(train_loader)}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.to(device))
            val_loss += criterion(outputs, labels.to(device)).item()

    # Save the model with the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

# Load the best model for testing
model.load_state_dict(torch.load('best_model.pth'))

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

# Compute evaluation metrics
cm = confusion_matrix(true_labels, predictions)
accuracy = accuracy_score(true_labels, predictions)
error_rate = 1 - accuracy
recall = recall_score(true_labels, predictions, average='weighted')
precision = precision_score(true_labels, predictions, average='weighted')

print("Confusion Matrix:")
print(cm)
print(f"Error Rate: {error_rate * 100:.2f}%")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")





