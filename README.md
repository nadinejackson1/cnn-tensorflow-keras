# 100DaysOfML - Day 14: Convolutional Neural Networks (CNNs) with TensorFlow and Keras

This project explores Convolutional Neural Networks (CNNs), a popular type of deep learning model, particularly for image recognition tasks. We implement a CNN classifier using TensorFlow and Keras, and apply it to the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Getting Started

### Prerequisites

    Python 3.x
    TensorFlow
    Keras
    Matplotlib
    NumPy
    scikit-learn

### Installing

In your terminal, run the following command to install the required libraries:

    pip install tensorflow keras matplotlib numpy scikit-learn

### Project Structure

- Load and preprocess the CIFAR-10 dataset: We load the dataset, normalize the images, and convert the labels to one-hot vectors for training.
- Create the CNN model: We create a CNN model using Keras with multiple convolutional and pooling layers, followed by dense layers for classification.
- Train the model: We train the model using the training dataset and validate it on the test dataset.
- Visualize training progress: We plot the training and validation accuracy over time to visualize the model's progress during training.
- Evaluate the model: We evaluate the model's performance on the test dataset by calculating the accuracy and generating a confusion matrix.
- Unit test: We provide a simple unit test to ensure the model's accuracy is above a specified threshold.

### Running the Project

- Clone the repository or download the project files.
- Open the project in your preferred Python IDE or run it in a Jupyter Notebook.
- Execute the code cells in the provided order, from loading the dataset to running the unit test.

### Built With

  **TensorFlow** - The machine learning framework used  
  **Keras** - The deep learning library used for building the CNN model  
  **CIFAR-10** - The image classification dataset used  
  **Matplotlib** - The library used for generating plots  
  **NumPy** - The library used for numerical computations  
  **scikit-learn** - The library used for evaluation metrics  
