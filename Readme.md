# Skin Disease Predictor

## Project Overview

This project is a Skin Disease Predictor, developed using a Convolutional Neural Network (CNN) to classify skin conditions based on uploaded images. The project uses PyTorch as the main deep learning framework, alongside Streamlit for the user interface. The model was trained on synthetic data generated for demonstration purposes. It includes several convolutional layers, max-pooling layers, batch normalization, and fully connected layers to achieve an end-to-end classification.

## File Structure

- **CODE_FILES Directory**: Contains the main project files:
  - `SkinDiseasePredictor.py`: Main Python script defining the CNN model class (`SkinDiseasePredictor`), training function, and dataset loading.
  - `skin_disease_predictor.pth`: Trained PyTorch model weights.
  - `pages/`: Contains additional pages for Streamlit, such as feedback and information pages (`1_Feedback_and_Support.py` and `2_More_information_page.py`).
  - `Filtered_data/`: Contains preprocessed images and labels.
  - `data_preprocessing.ipynb`, `data_to_numpy.ipynb`, etc.: Notebooks used for data preprocessing and conversion to suitable formats.

## Requirements

The following Python libraries are required to run this project:

- `torch`: The main library used for building and training the CNN model.
- `torchvision`: Used for transforming the input images to tensors.
- `numpy`: For data manipulation and generating synthetic data.
- `streamlit`: For building the user interface.
- `PIL` (from `Pillow`): For image manipulation.
- `opencv-python`: Used for live image capture if required.

## Model Architecture

The model architecture consists of the following components:

- **Convolutional Layers**: Five convolutional layers with increasing filter sizes (64, 128, 256).
- **Batch Normalization**: Added after each convolution layer to stabilize and accelerate training.
- **Max Pooling Layers**: Pooling layers after each convolutional block to reduce spatial dimensions.
- **Fully Connected Layers**: Three fully connected layers with dropout to prevent overfitting.
- **Activation Functions**: ReLU activation for all layers and softmax activation for the final output layer to predict 23 classes.

## Training

- **Training Data**: The training data consists of synthetic images generated using numpy for demonstration purposes. Images were randomly initialized with values from 0 to 255.
- **Training Function**: The model is trained using the `train_model()` function, which performs forward propagation, backpropagation, and updates weights using Adam optimizer.
- **Normalization**: Images are normalized by dividing pixel values by 255.0 to bring them to the range [0, 1].

## Running the Application

To run the project locally:

1. Clone or download the repository.
2. Make sure all dependencies are installed. You can use the following command to install them:
   ```sh
   pip install torch torchvision numpy streamlit pillow opencv-python
