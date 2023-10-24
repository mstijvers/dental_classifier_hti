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