import time
import streamlit as st
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the model class (matching what you used to train)
class SkinDiseasePredictor(nn.Module):
    def __init__(self):
        super(SkinDiseasePredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding='same')
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 13 * 22, 4096)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 23)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x

# Set the page configuration
st.set_page_config(page_icon="💊", layout="wide", page_title="Skin Disease Predictor")

# Title and subheader
st.title("Skin Disease Predictor🩺🥼💉")
st.subheader("Upload an image of the skin condition 📷", anchor=None)

# Allow user to upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load the trained model
PATH = "/Users/shreyassawant/mydrive/Shreyus_workspace/Semester_V/IBM_Datathon/skin_disease_predictor.pth"
our_model = SkinDiseasePredictor()
our_model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
our_model.eval()

# Placeholder for results
if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    
    # Display the image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Image uploaded and displayed successfully!")

    # Convert image to tensor
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Button for prediction
    if st.button("Predict"):
        with torch.no_grad():
            prediction = our_model(image_tensor)
            predicted_class = torch.argmax(prediction, dim=1).item()
            st.success(f"Predicted Skin Condition: *Class {predicted_class}*")
else:
    st.warning("Please upload an image to proceed with prediction.")

# Button for live capture of image.
if st.button("Live capture 📸"):
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)
    # Loop for a 3 second delay.
    for waqt in range(3):
        st.write(3 - waqt, "..........")
        time.sleep(1)
    # Capture the image from camera
    result, image = cam.read()
    if result:
        st.image(image, caption='Captured Image.', use_column_width=True)
    else:
        st.error("Failed to capture image. Please try again.")
    cam.release()

