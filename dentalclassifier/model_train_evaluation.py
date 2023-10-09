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

# 定义类别标签
class_labels = ["gingivitis", "hypodontia", "discoloration", "caries", "calculus", "healthyteeth"]

# 创建空列表来存储图像和标签对
data = []
labels = []

# 数据文件夹的相对路径
data_folder = "Dentaldata"

# 修改加载图像的部分代码
for class_label in class_labels:
    class_folder = os.path.join(data_folder, class_label)
    for filename in os.listdir(class_folder):
        if filename.endswith(".jpg"):  # 假设图像文件都以 .jpg 结尾
            img_path = os.path.join(class_folder, filename)
            try:
                with Image.open(img_path) as img:  # 使用 with 语句加载图像
                    if img is not None and len(np.array(img).shape) == 3:
                        data.append(img)
                        labels.append(class_labels.index(class_label))  # 使用类别的索引作为标签
                        print(f"Loaded image: {img_path}, Label: {class_labels.index(class_label)}")
                    else:
                        print(f"Skipped invalid image: {img_path}")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                

# 将数据转换为NumPy数组
data = np.array(data, dtype=object)
labels = np.array(labels)

# 添加以下代码来检查数据
print(f"Total images: {len(data)}")
print(f"Total labels: {len(labels)}")
for class_label in class_labels:
    class_count = (labels == class_labels.index(class_label)).sum()
    print(f"Total images in class '{class_label}': {class_count}")

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 创建模型
num_classes = 6  # 假设有6个类别
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
train_dataset = TensorDataset(X_train_tensor, torch.tensor(y_train))
test_dataset = TensorDataset(X_test_tensor, torch.tensor(y_test))

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5  # 根据需要选择训练周期数
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
    
    # 打印每个周期的损失
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# 保存模型的权重和结构
torch.save(model.state_dict(), 'with_healthyteeth_model.pth')

# 保存类别标签映射，以便在推理时使用
import json
with open('class_labels.json', 'w') as f:
    json.dump(class_labels, f)

# 在测试集上评估模型
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
