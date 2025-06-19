# app.py
import streamlit as st
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os





# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
@st.cache_resource
def load_model(checkpoint_path):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# Define preprocessing transform
def preprocess_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load and preprocess image
def load_and_preprocess(image):
    # Convert Streamlit uploaded image (PIL) to RGB
    image = image.convert('RGB')
    
    # Preprocess for model input
    transform = preprocess_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Preprocess for visualization (normalize to 0-1)
    rgb_img = np.array(image.resize((224, 224))) / 255.0
    
    return input_tensor, rgb_img

# Generate Grad-CAM visualization
def generate_gradcam(model, target_layers, input_tensor, rgb_img, class_idx=None):
    cam = GradCAM(model=model, target_layers=target_layers)
    
    if class_idx is None:
        with torch.no_grad():
            output = model(input_tensor)
            class_idx = torch.argmax(output).item()
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=None, aug_smooth=True, eigen_smooth=True)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return visualization, class_idx

# Generate multi-target Grad-CAM
def multi_target_gradcam(model, target_layers, input_tensor, rgb_img):
    cam = GradCAM(model=model, target_layers=target_layers)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    fig, axs = plt.subplots(1, 5, figsize=(20,20))
    axs[0].imshow(rgb_img)
    axs[0].set_title("Original Image", fontsize=30)
    axs[0].axis('off')
    
    class_names = {0: 'glioma', 1: 'meningioma', 2: 'no tumor', 3: 'pituitary'}
    for i, class_name in class_names.items():
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(i)], aug_smooth=True, eigen_smooth=True)[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        axs[i+1].imshow(visualization)
        axs[i+1].set_title(f"{class_name}\nprob: {probabilities[i]:.2f}", fontsize=30)
        axs[i+1].axis('off')
    
    plt.tight_layout()
    return fig

# Compare different CAM methods
def compare_methods(model, target_layers, input_tensor, rgb_img):
    methods = {
        "GradCAM": GradCAM,
        "GradCAM++": GradCAMPlusPlus,
        "XGradCAM": XGradCAM,
        "EigenCAM": EigenCAM
    }
    
    with torch.no_grad():
        output = model(input_tensor)
        class_idx = torch.argmax(output).item()
    
    
    fig, axs = plt.subplots(1, len(methods)+1)
    axs[0].imshow(rgb_img)
    axs[0].set_title("\n\nOriginal Image")
    axs[0].axis('off')
    
    class_names = {0: 'glioma', 1: 'meningioma', 2: 'no tumor', 3: 'pituitary'}
    for i, (name, method_class) in enumerate(methods.items()):
        cam = method_class(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)], aug_smooth=True)[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        axs[i+1].imshow(visualization)
        axs[i+1].set_title(f"\n\n{name}")
        axs[i+1].axis('off')
    
    plt.suptitle(f"Comparison for {class_names[class_idx]} tumor")
    plt.tight_layout()
    return fig

# Streamlit app
st.title("Brain Tumor Classification with Grad-CAM Visualization")

# Model path (update this to your model path)
model_path = "brain_tumor_resnet18.pth"  # Update with your model path
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please provide the correct path.")
    st.stop()

# Load model
model = load_model(model_path)
target_layers = [model.layer4[-1].conv2]

# File uploader
uploaded_file = st.file_uploader("Upload a brain MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    input_tensor, rgb_img = load_and_preprocess(image)
    
    # Generate Grad-CAM for predicted class
    st.subheader("Grad-CAM for Predicted Class")
    cam_image, class_idx = generate_gradcam(model, target_layers, input_tensor, rgb_img)
    class_names = {0: 'glioma', 1: 'meningioma', 2: 'no tumor', 3: 'pituitary'}
    st.image(cam_image, caption=f"Grad-CAM for {class_names[class_idx]} tumor", use_column_width=True)
    st.write(f"Predicted class: {class_names[class_idx]}")
    
    # Multi-target Grad-CAM
    st.subheader("Multi-Target Grad-CAM for All Classes")
    multi_fig = multi_target_gradcam(model, target_layers, input_tensor, rgb_img)
    st.pyplot(multi_fig)
    
    # Compare CAM methods
    st.subheader("Comparison of Different CAM Methods")
    compare_fig = compare_methods(model, target_layers, input_tensor, rgb_img)
    st.pyplot(compare_fig)