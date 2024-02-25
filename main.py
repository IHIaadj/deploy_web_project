import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

def predict(image):
    # Define the image transformation
    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformations to the image
    image = transformation(image).unsqueeze(0)  # Add batch dimension

    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Convert the predicted class index to a label
    class_idx = predicted.item()
    labels = ["class_name_1", "class_name_2", "..."]  # Replace with actual class names
    predicted_label = labels[class_idx]

    return predicted_label

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()  # Set the model to evaluation mode

st.title("Image Classification with Deep Learning (PyTorch)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(image)
    st.write(f'Prediction: {label}')
