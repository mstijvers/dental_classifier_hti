import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
            xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'],
            yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'])

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


# model loss
epochs = list(range(1, 21))
loss = [1.4408316430838213, 0.680382578269295, 0.43636820198077225, 0.24721717881038785, 0.17839390984938844,
        0.15188998164123166, 0.12588839866625873, 0.12234973330202553, 0.11907313137983336, 0.10706081269207694,
        0.10475502631869978, 0.10395229643757414, 0.09902277816613407, 0.09829039585209735, 0.09808958369641281,
        0.09411195520837994, 0.09387837909162045, 0.09414681163616478, 0.09217214356961093, 0.0919098891975844]
learning_rates = [0.001, 0.001, 0.0005, 0.0005, 0.0005, 0.00025, 0.00025, 0.00025, 0.000125, 0.000125, 0.000125,
                  6.25e-05, 6.25e-05, 6.25e-05, 3.125e-05, 3.125e-05, 3.125e-05, 1.5625e-05, 1.5625e-05, 1.5625e-05]

plt.figure(figsize=(10, 5))
plt.plot(epochs, loss, marker='o', linestyle='-', color='b', label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.xticks(epochs)
plt.grid(True)

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




