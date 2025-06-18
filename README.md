# Brain Tumor Classification with Grad-CAM Visualization

This Streamlit application enables users to upload brain MRI images and perform tumor classification using a pre-trained ResNet18 model, with visualizations provided by various Grad-CAM techniques. The app supports four tumor classes: glioma, meningioma, no tumor, and pituitary.

## Features

- **Image Upload**: Upload brain MRI images in JPG or PNG format.
- **Model Prediction**: Uses a pre-trained ResNet18 model to classify the uploaded image into one of four tumor classes.
- **Grad-CAM Visualization**: Displays Grad-CAM heatmap for the predicted class, highlighting important regions for the model's decision.
- **Multi-Target Grad-CAM**: Visualizes Grad-CAM heatmaps for all four classes alongside their predicted probabilities.
- **CAM Methods Comparison**: Compares visualizations from different CAM methods (GradCAM, GradCAM++, XGradCAM, EigenCAM) for the predicted class.
- **Responsive Design**: Adapts visualization sizes based on the browser window width.

## Requirements

- Python 3.8+
- Streamlit
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- PIL (Pillow)
- torchvision
- pytorch-grad-cam
- streamlit-js-eval

Install dependencies using:

```bash
pip install streamlit torch opencv-python numpy matplotlib pillow torchvision pytorch-grad-cam streamlit-js-eval
```

## Setup

1. **Model File**:
   - Place your pre-trained ResNet18 model file (`brain_tumor_resnet18.pth`) in the project directory.
   - Update the `model_path` variable in `app.py` if the model file is located elsewhere.

2. **Directory Structure**:
   ```
   project_directory/
   ├── app.py
   ├── brain_tumor_resnet18.pth
   └── README.md
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the app in your browser (typically at `http://localhost:8501`).
2. Upload a brain MRI image (JPG or PNG).
3. View the results:
   - **Uploaded Image**: Displays the original MRI image.
   - **Grad-CAM for Predicted Class**: Shows the Grad-CAM heatmap for the model's predicted tumor class.
   - **Multi-Target Grad-CAM**: Displays heatmaps for all four classes with their respective probabilities.
   - **Comparison of Different CAM Methods**: Compares visualizations from GradCAM, GradCAM++, XGradCAM, and EigenCAM.

## Code Overview

- **Model Loading**: Loads a pre-trained ResNet18 model with a modified fully connected layer for 4-class classification.
- **Image Preprocessing**: Resizes images to 224x224, normalizes them, and converts them to tensors for model input.
- **Grad-CAM Implementation**: Uses the `pytorch_grad_cam` library to generate heatmaps for model interpretability.
- **Visualization**:
  - Single-class Grad-CAM for the predicted class.
  - Multi-class Grad-CAM for all classes with probability scores.
  - Comparison of different CAM methods for the predicted class.
- **Streamlit Interface**: Provides an interactive UI for image upload and result visualization.

## Notes

- Ensure the model file path is correct; otherwise, the app will stop with an error.
- The app uses the last convolutional layer (`layer4[-1].conv2`) of ResNet18 for Grad-CAM visualizations.
- Visualizations are normalized for display and use RGB format.
- The app dynamically adjusts figure sizes based on browser window width for better viewing.
- CUDA is used if available; otherwise, it falls back to CPU.

## Limitations

- The model must be trained and saved as `brain_tumor_resnet18.pth` in the specified format.
- Only JPG and PNG images are supported.
- The app assumes the input images are brain MRIs; other image types may lead to unreliable predictions.
- Visualization quality depends on the input image resolution and model performance.

## Troubleshooting

- **Model Not Found**: Verify the `brain_tumor_resnet18.pth` file exists in the specified path.
- **Dependency Issues**: Ensure all required packages are installed with compatible versions.
- **CUDA Errors**: If GPU is unavailable or misconfigured, the app automatically uses CPU.
- **Visualization Issues**: Check browser compatibility or adjust the `page_width` calculation if figures appear misaligned.

