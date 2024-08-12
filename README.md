# Image Classifier

Name: Harshika Jain   
Company: CODTECH IT SOLUTIONS  
ID: CT6AIO7  
Domain: Artificial Intelligence  
Duration: 20 June 2024 to 20 August 2024  

## Overview

This project is an image classification application using a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The application uses TensorFlow and Keras to build, train, and evaluate the CNN model, achieving high accuracy in classifying images into one of 10 classes.

## Features

- **Convolutional Neural Network (CNN):** The model is built using multiple convolutional layers to extract features from images, followed by fully connected layers for classification.
- **Real-time Image Classification:** After training, the model can classify new images in real-time.
- **Training and Validation:** The model is trained on the CIFAR-10 training set and validated on the test set, with accuracy and loss plots generated after training.
  
## Dataset:  

**CIFAR-10 Dataset:** The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. The classes include:  
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck
  
## Technology Used  
- **Python 3.x** (Core language for backend and data processing)
- **TensorFlow / Keras** (Deep learning frameworks for building and training the CNN model)
- **Matplotlib** (For plotting training and validation accuracy)
- **NumPy** (For numerical operations)
- **CIFAR-10 Dataset** (Standard dataset for image classification tasks)

## Setup and Installation

### Prerequisites

- Python 3.x
- pip (Python package installer)
- Virtual environment (optional but recommended)

### Step 1: Clone the Repository

```
git clone https://github.com/Harshikajain23/CODTECH-Internship-Task2.git
cd CODTECH-Internship-Task2
```


### Step 2: Install Dependencies

```
pip install tensorflow matplotlib numpy
```

### Step 3: Run the Program
```
python classify.py
```

## Usage

- After running the program, the model will be trained on the CIFAR-10 dataset for 10 epochs.
- The training history, including accuracy and validation accuracy, will be plotted.
- The first image in the test set will be displayed with its predicted and actual labels.


## License

This project is licensed under the MIT License - see the LICENSE file for details.



![snap 1](https://github.com/user-attachments/assets/f92e4221-ad02-4d09-9078-94ffc93128b0)

![snap 2](https://github.com/user-attachments/assets/0fbe9be9-96b2-4395-b117-5e7d31e8baf0)


