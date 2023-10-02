
import torch.nn as nn
from flask import Flask, render_template, request, redirect, url_for
import os, sys
from torchvision import transforms
import torch
from PIL import Image

torch.no_grad()

sys.path.append(os.getcwd())

from dentalclassifier.modules import tanden

app = Flask(__name__, template_folder="templates")

def load_model(model_pth, num_classes = 5):
    model = tanden.SimpleCNN(num_classes)

    model.load_state_dict(torch.load(model_pth))
    model.eval()
    return model

def load_transform():
    return transforms.Compose([
        transforms.ToTensor(),  # Convert the PIL Image to a PyTorch tensor
        transforms.Resize((224, 224)),      # Resize to the size expected by your model (e.g., 224x224 for many pre-trained models)
        # transforms.Normalize(mean, std)   # If your model expects normalized data, add this line with the appropriate mean and std
    ])


@app.route('/')
def home():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    result = None

    print(f"{request.method}")

    if request.method == "POST":
        print(f"{request.files}")
        if "image" in request.files:
            uploaded_image = request.files["image"]
            model = load_model(model_pth="dental_classifier.pth")

            img = Image.open(uploaded_image)
            print(img.size)
            
            if uploaded_image.filename != "":
                # Save the uploaded image
                # image_path = os.path.join(os.path.join(os.getcwd(), "static/"), uploaded_image.filename)
                image_path = uploaded_image
                # uploaded_image.save(image_path)

                transform = load_transform()

                # Preprocess the image
                img = Image.open(image_path)
                img = transform(img)
                img = img.unsqueeze(0)  # Add a batch dimension (single image)

                print(img.shape)

                print(model)

                # Pass the image through the model
                output = model(img)

                print(output)

                # Apply softmax to obtain class probabilities
                softmax         = nn.Softmax(dim=1)
                probabilities   = softmax(output)
                class_labels    = ["gingivitis", "hypodontia", "discoloration", "caries", "calculus"]
                predicted_class = class_labels[torch.argmax(probabilities)]
                conf_score      = torch.max(probabilities).item()

                # Prepare the result message
                result = f"Predicted Class: {predicted_class}, Confidence Score: {conf_score:.2f}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
