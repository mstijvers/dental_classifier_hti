import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from tanden import SimpleCNN
import torch
import torch.nn as nn

from model_train_evaluation import train_loader, val_loader,criterion

# Create an instance of the model with the same architecture as in code 1
num_classes = 6  # Adjust the number of classes to match your dataset
model = SimpleCNN(num_classes)


# model loss
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()

avg_test_loss = test_loss / len(val_loader)


plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.title('Training and Validation Loss Over Epochs')
plt.show()


# confusion matrix

confusion_matrix = np.array([[396, 4, 6, 5, 65, 3],
                            [6, 43, 1, 8, 5, 0],
                            [9, 2, 249, 84, 1, 11],
                            [18, 3, 124, 340, 6, 3],
                            [106, 0, 1, 3, 135, 0],
                            [4, 0, 8, 0, 0, 190]])


plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)

# Create a heatmap with annotations and an adjusted color map
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlGnBu', cbar=False,
            xticklabels=["gingivitis", "hypodontia", "discoloration", "caries", "calculus", "healthyteeth"],
            yticklabels=["gingivitis", "hypodontia", "discoloration", "caries", "calculus", "healthyteeth"])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Improve readability by rotating x and y tick labels
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Adjust spacing to prevent cutoff of labels
plt.tight_layout()

plt.savefig('Confusion_Matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# learning rate
plt.twinx()  
plt.plot(epochs, learning_rates, marker='s', linestyle='--', color='r', label='Learning Rate')
plt.ylabel('Learning Rate')

plt.legend(loc='upper left')
plt.legend(loc='upper right')


plt.savefig('Model_Loss_Plot.png')
plt.show()



# use a table to list rates

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

metrics = ['Error Rate', 'Accuracy', 'Recall', 'Precision']
values = [26.43, 73.57, 73.57, 73.68]

# Create a DataFrame from the metrics and values
data = {'Metric': metrics, 'Value': values}
df = pd.DataFrame(data)

width, title_height = 400, 30  # Adjust the width to your preference
cell_height = 30  # Adjust the cell height to your preference
height = title_height + len(metrics) * cell_height

image = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

babyblue_color = (173, 216, 230)

title_font = ImageFont.load_default()
draw.rectangle([0, 0, width, title_height], fill=babyblue_color)
draw.text((10, 5), "Performance Metrics", fill='white', font=title_font)

for i, row in df.iterrows():
    y_start = title_height + i * cell_height
    y_end = title_height + (i + 1) * cell_height
    draw.rectangle([0, y_start, width, y_end], outline='black', width=1)
    draw.text((10, y_start + 5), row['Metric'], fill='black', font=font)
    formatted_value = f"{row['Value']:.2f}%"
    draw.text((width / 2 + 10, y_start + 5), formatted_value, fill='black', font=font)

image.save('styled_model_performance_metrics.png')
image.show()




