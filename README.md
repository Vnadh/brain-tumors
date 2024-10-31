# Brain Tumor Classification with Explainability
This Streamlit application classifies brain MRI images into four categories of brain tumors using a Convolutional Neural Network (CNN) model. To enhance interpretability, it utilizes the LIME (Local Interpretable Model-agnostic Explanations) technique to visually highlight the image regions that most influence the model’s prediction.
## Project Overview
Brain tumor classification can aid medical professionals in diagnosing and selecting appropriate treatments. This project applies deep learning to classify MRI brain images into four categories:

    * Glioma
    * Meningioma
    * No Tumor
    * Pituitary
To explain the model's decisions, it leverages LIME to show which parts of the image the model considers most influential.
## Features
**Image Classification:** Predicts brain tumor type from uploaded MRI images.
**Explainability:** LIME visualization highlights influential regions of the image that contribute to the model’s prediction.
**Confidence Score:** Displays the model's confidence level for each prediction.
## Prerequisites
Ensure you have the following installed:

- Python 3.7+
- TensorFlow
- Streamlit
- LIME
- Matplotlib
### Model Architecture
The CNN model is built with three convolutional layers followed by max-pooling, flattening, and dense layers for classification. It has been trained on brain MRI images for multi-class classification.

### Explainability with LIME
LIME (Local Interpretable Model-agnostic Explanations) provides model interpretability by showing which image regions the model focuses on when making predictions. The app overlays a mask on the MRI image, highlighting the influential features that led to the model’s decision.
