import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Load the trained model
model_path = 'brain_tumor_model.h5'  # Update path if needed
model = load_model(model_path)

# Define class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Streamlit app title and description
st.title("Brain Tumor Classification with Explainability")
st.write("This app classifies brain MRI images into four categories and provides model explainability using LIME.")

# File uploader for the MRI image
uploaded_file = st.file_uploader("Upload an MRI image (in .jpg or .png format)", type=["jpg", "png"])

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    img = load_img(uploaded_file, target_size=(180, 180))  # Resize to match model's input
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Display the uploaded image
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    confidence = np.max(predictions) * 100  # Confidence in percentage

    # Display the prediction result
    st.write(f"### Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Lime explanation
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_array[0].astype('double'),
                                             model.predict,
                                             top_labels=4,
                                             hide_color=0,
                                             num_samples=1000)

    # Extract LIME explanation mask and image
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=True,
                                                num_features=5,
                                                hide_rest=True)

    # Plot the LIME explanation
    fig, ax = plt.subplots()
    ax.imshow(mark_boundaries(temp, mask))
    ax.set_title(f"LIME Explanation - Predicted Class: {predicted_class}")
    st.pyplot(fig)
