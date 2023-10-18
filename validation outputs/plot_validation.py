import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# confusion matrix
confusion_matrix = np.array([[332, 1, 12, 10, 123],
                             [11, 5, 49, 5, 5],
                             [3, 3, 236, 102, 3],
                             [91, 5, 3, 137, 346],
                             [2, 11, 56, 1, 2]])


plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2) 


sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'],
            yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.savefig('Confusion Matrix.png')
plt.show()




# model loss
epochs = list(range(1, 11))
loss_values = [1.4483, 0.7281, 0.5552, 0.4273, 0.3248, 0.2334, 0.1898, 0.1582, 0.1429, 0.1418]

plt.figure(figsize=(8, 6))
plt.plot(epochs, loss_values, marker='o', linestyle='-')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.xticks(epochs)

for i, loss in enumerate(loss_values):
    plt.annotate(f'{loss:.4f}', (epochs[i], loss), textcoords="offset points", xytext=(0,10), ha='center')

plt.savefig('Model_Loss_Plot.png')
plt.show()



# use a table to list rates

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

metrics = ['Error Rate', 'Accuracy', 'Recall', 'Precision']
values = [28.49, 71.51, 71.51, 73.17]  


width, height = 800, 200
image = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

babyblue_color = (173, 216, 230)  

title_font = ImageFont.load_default()
title_height = 30
draw.rectangle([0, 0, width, title_height], fill=babyblue_color)  
draw.text((10, 5), "Performance Metrics", fill='white', font=title_font)


cell_width = width / 2
cell_height = (height - title_height) / len(metrics)
for i, metric in enumerate(metrics):
    y_start = title_height + i * cell_height
    y_end = title_height + (i + 1) * cell_height
    draw.rectangle([0, y_start, cell_width, y_end], outline='black', width=1)
    draw.text((10, y_start + 5), metric, fill='black', font=font)

for i, value in enumerate(values):
    y_start = title_height + i * cell_height
    y_end = title_height + (i + 1) * cell_height
    draw.rectangle([cell_width, y_start, width, y_end], outline='black', width=1)
    formatted_value = f"{value:.2f}%"
    draw.text((cell_width + 10, y_start + 5), formatted_value, fill='black', font=font)

image.save('styled_model_performance_metrics.png')
image.show()



