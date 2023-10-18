from PIL import Image
from functools import partial
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from tanden import SimpleCNN
import torch
import torchvision.transforms as transforms
import torch.nn as nn

# what does this mean? @silke??
visualized_class = 4


image_path = "static/images/cropped_mouth.jpg"
model_path = "../dental_classifier.pth"
img = Image.open(image_path)
org_width, org_height = img.size

def fetch_model(model_path) -> nn.Module:
    """
    Imports model weights from file and returns the model set to eval mode.
    """
    num_classes = 5
    model = SimpleCNN(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def process_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor()
    ])
    return transform(img)

def transform_classifier_to_lime(img: torch.Tensor) -> np.array:
    """Takes one image and transforms it to meet the lime requirements"""
    red_matrix, green_matrix, blue_matrix = img
    rows, columns = img[0].shape
    transformed_img = []
    for index_row in range(rows):
        transformed_row = []
        for index_column in range(columns):
            pixel = [
                red_matrix[index_row][index_column],
                green_matrix[index_row][index_column],
                blue_matrix[index_row][index_column]
            ]
            transformed_row.append(pixel)
        transformed_img.append(transformed_row)
    return np.array(transformed_img)

def transform_lime_to_classifier(img: torch.Tensor) -> torch.Tensor:
    """Takes image from lime and transforms it to classifier requirements"""
    red_matrix, green_matrix, blue_matrix = [], [] , []

    for rows in img:
        red_row = []
        green_row = []
        blue_row = []
        for pixel in rows:
            red, green, blue = pixel
            red_row.append(red)
            green_row.append(green)
            blue_row.append(blue)
        red_matrix.append(red_row)
        green_matrix.append(green_row)
        blue_matrix.append(blue_row)
    return torch.Tensor([red_matrix, green_matrix, blue_matrix]).unsqueeze(0)

def predict_fn(images, transform_required=False):
    """Very hacky code to transform image from lime to something the model can process, 
    returning probabilities that lime can work with."""
    model = fetch_model(model_path)
    if images.ndim == 3:
        # if just one image is supplied
        images = np.expand_dims(images, axis=0)
    if transform_required:
        first_run = True
        for image in images:
            if first_run:
                transformed_images = transform_lime_to_classifier(
                    image
                )
            else:
                transformed_images = torch.cat(
                        [
                        transformed_images, 
                        transform_lime_to_classifier(image)
                    ]
                )
            first_run = False
        images = transformed_images
    with torch.no_grad():
        softmax = nn.Softmax(dim=1)
        outputs = []
        for image in images:
            output = model(image.unsqueeze(0))
            
            probabilities = softmax(output)
            outputs.append(probabilities[0].numpy())
    return outputs

predict_fn_lime = partial(predict_fn, transform_required=True)

def overlay_explainability_layer(explanation):
    temp, mask = explanation.get_image_and_mask(visualized_class, positive_only=True, num_features=5, hide_rest=False)
    plt.imsave("static/images/analyzed_teeth.jpg", mark_boundaries(temp+ 0.3, mask, color=(0.5294117647058824, 0.09803921568627451, 0.19607843137254902)))
    # Open the saved image
    overlayed_image = Image.open("static/images/analyzed_teeth.jpg")
    # Resize the overlayed image back to its original dimensions
    overlayed_image = overlayed_image.resize((org_width, org_height))
    # Save the resized overlayed image
    overlayed_image.save("static/images/analyzed_teeth_resized.jpg")


def main(img):
    img = process_image(img)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        transform_classifier_to_lime(img), 
        classifier_fn=predict_fn_lime, 
        labels = ["gingivitis", "hypodontia", "discoloration", "caries", "calculus", "healthy teeth"],
        top_labels=5, 
        hide_color=0, 
        num_samples=10
    )
    overlay_explainability_layer(explanation)

if __name__ == "__main__":
    main(img)

