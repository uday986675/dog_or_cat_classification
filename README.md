Dog vs Cat Classification

This project is a deep learning-based image classification system that distinguishes between images of dogs and cats. It leverages MobileNetV2 as the backbone model to achieve high accuracy while keeping the model lightweight and efficient.

Features

Classifies images into Dog or Cat categories.

Uses transfer learning with MobileNetV2 for fast and efficient training.

Supports GPU acceleration for faster training and inference.

Provides preprocessing and augmentation for better generalization.

Model Architecture

Base Model: MobileNetV2 (pretrained on ImageNet)

Custom Layers: Global Average Pooling + Dense layers for classification

Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy

Dataset

The dataset consists of labeled images of dogs and cats.

Note: The dataset is not included due to its large size.

You can use the Kaggle Dogs vs Cats dataset
 or any similar dataset.

The images should be organized in the following directory structure:

dataset/
├── train/
│   ├── dogs/
│   └── cats/
├── validation/
│   ├── dogs/
│   └── cats/

Installation
git clone <your-repo-url>
cd dog-cat-classification
pip install -r requirements.txt


Requirements:

Python 3.8+

TensorFlow 2.x

NumPy

Matplotlib (optional, for visualizations)

Usage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model('dog_cat_model.h5')

# Load and preprocess image
img = image.load_img('path_to_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("Predicted: Dog")
else:
    print("Predicted: Cat")

Training

Preprocess images: resize to 224x224 and normalize pixel values.

Use ImageDataGenerator for augmentation (rotation, flip, zoom).

Fine-tune the last few layers of MobileNetV2.

Train using binary crossentropy loss and monitor validation accuracy.

Results

Achieves high accuracy on validation set (~90%+ depending on dataset size and training).

Fast inference on new images thanks to MobileNetV2.

Future Improvements

Experiment with other pretrained architectures like EfficientNet or ResNet.

Add more data augmentation techniques.

Deploy the model as a web or mobile application.
