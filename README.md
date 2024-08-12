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
- **CIFAR-10 Dataset:** The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. The classes include:  
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
git clone https://github.com/Harshikajain23/CODTECH-Internship-Task1.git
cd CODTECH-Internship-Task1
```

### Step 2: Set Up Virtual Environment (Optional)

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install Dependencies

```
pip install -r requirements.txt 
```

### Step 4: Download NLTK Data

In your Python environment, run the following script to download the necessary NLTK data files:

```
import nltk
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
```

### Step 5: Run the Application

```
python app.py
```
This will start the Flask development server. By default, it runs on http://127.0.0.1:5000.  
Now you can access the project by navigating to http://127.0.0.1:5000 in your web browser.


## Usage

- After running the program, the model will be trained on the CIFAR-10 dataset for 10 epochs.
- The training history, including accuracy and validation accuracy, will be plotted.
- The first image in the test set will be displayed with its predicted and actual labels.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

![screen shot 1](https://github.com/Harshikajain23/CODTECH-Internship-Task1/assets/129208900/903c6f0d-3544-4844-ba35-d4c4076b4a66)

![screen shot 2](https://github.com/Harshikajain23/CODTECH-Internship-Task1/assets/129208900/d9b6189a-d1d9-47f4-b465-ff6a28498001)
