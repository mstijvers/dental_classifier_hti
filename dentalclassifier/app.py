import torch.nn as nn
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import os
from torchvision import transforms
import torch
from PIL import Image
from tanden import SimpleCNN
from camera import VideoCamera
import cv2
import dlib
import math
import json
import numpy as np
from lime.lime_image import LimeImageExplainer

app = Flask(__name__, template_folder="templates")

# video stream
video_stream = VideoCamera()

# Load the class labels
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Create an instance of the model with the same architecture as in code 1
num_classes = len(class_labels)  # Adjust the number of classes to match your dataset
model = SimpleCNN(num_classes)

# Load the pre-trained weights into the model

model.load_state_dict(torch.load('with_healthyteeth_model.pth'))



# Set the model to evaluation mode
model.eval()

# Define a transformation for preprocessing the uploaded image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3]),
    # Add other transformations as needed (e.g., normalization)
])


# start server and render index html
@app.route('/')
def home():
    # Redirect to the index.html file
    return render_template("index.html")


# innitialize camera
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# get video view
@app.route('/video_feed')
def video_feed():
    return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# reset video view if when button is pressed
@app.route('/reset_camera', methods=['POST'])
def reset_camera():
    if request.method == 'POST':
        video_stream.reset_camera()
    return render_template("index.html")


# define the variables in the beginning
result = None
formatted_confidence_score = None
predicted_class = None
confidence_score = None
explanation_overlay=None

# classify image
@app.route("/classify", methods=["POST"])
def classify():
    global result, formatted_confidence_score, predicted_class, confidence_score, explanation_overlay

    image_path = "static/images/cropped_mouth.jpg"  # Path to the cropped_mouth.jpg image

    if os.path.isfile(image_path):
        # Preprocess the image
        img = Image.open(image_path)
        img = transform(img)
        img = img.unsqueeze(0)  # Add a batch dimension (single image)

        # Pass the image through the model
        print(img.shape)
        with torch.no_grad():
            output = model(img)

        # Apply softmax to obtain class probabilities
        softmax = nn.Softmax(dim=1)
        probabilities = softmax(output)
        class_labels = ["gingivitis", "hypodontia", "discoloration", "caries", "calculus", "healtyteeth"]
        label_mapping = {label: idx for idx, label in enumerate(class_labels)}

        predicted_class = class_labels[torch.argmax(probabilities)]
        confidence_score = torch.max(probabilities).item()


        # Check if confidence_score is a valid number
        if confidence_score is not None and not math.isnan(confidence_score):
            # Convert confidence_score to a percentage
            confidence_score_percentage = confidence_score * 100
            formatted_confidence_score = f"{confidence_score_percentage:.0f}%"
        else:
            # Handle the case where confidence_score is not a valid number
            formatted_confidence_score = "N/A"

        #video wizard of oz'ing
        #predicted_class = "caries"

        # Prepare the result message
        result = f"Predicted Class: {predicted_class}, Confidence Score: {confidence_score:.2f}"


    print(predicted_class)
    return render_template("results.html",
                           pclass=predicted_class,
                           cscore=confidence_score,
                           #cscore_p= f"{78:.0f}%",
                           cscore_p=formatted_confidence_score,
                           )
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/home')
def firsthome():
    return render_template('home.html')

@app.route('/dentalcare')
def dentalcare():
    return render_template('dentalcare.html')


if __name__ == "__main__":
    app.run(port=8000, debug=True)
