# Cat vs Dog Classification with TensorFlow and Keras

## Overview

This project involves building a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images of cats and dogs. The model achieves an accuracy of 85%. The project includes a full analysis of the model's performance. The setup and execution can be performed in two different environments: Google Colab and Jupyter Notebook.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
  - [Google Colab](#google-colab)
  - [Jupyter Notebook](#jupyter-notebook)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Results](#results)
- [Confusion Matrix](#confusion-matrix)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

This project demonstrates the process of building and training a CNN to classify images of cats and dogs. The model is trained on the popular Dogs vs. Cats dataset. The dataset consists of 25,000 images of cats and dogs, and the goal is to classify each image correctly as either a cat or a dog.

## Dataset

The dataset used is the [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) dataset available on Kaggle. It contains 25,000 labeled images of cats and dogs.

## Environment Setup

### Google Colab

To run the project in Google Colab, follow these steps:

1. Upload your `kaggle.json` file to your Google Drive.
2. Mount your Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3. Install necessary libraries and set up Kaggle API:
    ```python
    !pip install kaggle
    !mkdir ~/.kaggle
    !cp /content/drive/My\ Drive/kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    ```
4. Download the dataset:
    ```python
    !kaggle competitions download -c dogs-vs-cats
    !unzip dogs-vs-cats.zip -d ./data/
    ```

### Jupyter Notebook

To run the project in a Jupyter Notebook:

1. Install the necessary libraries:
    ```bash
    pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn pydot graphviz
    ```
2. Download the dataset from Kaggle and unzip it in your working directory.

## Model Architecture

The model architecture is a Convolutional Neural Network (CNN) consisting of:

- Convolutional layers with ReLU activation
- MaxPooling layers
- Flatten layer
- Dense layers with ReLU activation
- Output layer with Sigmoid activation

## Training the Model

The model is trained using the following parameters:

- Optimizer: Adam
- Loss function: Binary Crossentropy
- Metrics: Accuracy
- Epochs: 10
- Batch size: 20

## Evaluating the Model

The model's performance is evaluated using accuracy and a confusion matrix. The accuracy achieved is 85%.

## Results

The model achieves an accuracy of 85% on the test set. Below is the code snippet used for training and evaluating the model:

```python
history = model.fit(
    train_generator, 
    steps_per_epoch=train_steps, 
    epochs=10, 
    validation_data=validation_generator, 
    validation_steps=validation_steps
)
```

## Confusion Matrix

A confusion matrix is used to visualize the performance of the model. It shows the number of true positive, true negative, false positive, and false negative predictions.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Assuming true_classes and predicted_classes are defined
cm = confusion_matrix(true_classes, predicted_classes)

# Set up the plot
plt.figure(figsize=(10, 8))

# Create a heatmap for the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)

# Add titles and labels
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Display the plot
plt.show()
```

## Conclusion

This project demonstrates the successful implementation of a CNN to classify images of cats and dogs, achieving an accuracy of 85%. The model analysis includes training and evaluation metrics, along with a confusion matrix to better understand the model's performance.

## References

- [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)

Feel free to explore and modify the code to improve the model's accuracy or to adapt it to other image classification tasks. Happy coding!
