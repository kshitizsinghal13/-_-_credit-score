import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('handwritten_character_recognition_model.keras')

# Load label mapping from the CSV file (or define it manually)
csv_file_path = 'english.csv'  # Update this path to your CSV file location
data = pd.read_csv(csv_file_path)
label_mapping = {idx: char for idx, char in enumerate(sorted(data['label'].unique()))}

# Function to preprocess a single image for prediction
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found at path: {img_path}")
    
    # Resize image to 28x28 pixels
    img = cv2.resize(img, (28, 28))
    img = img.reshape(-1, 28, 28, 1) / 255.0  # Normalize and reshape
    return img

# Function to predict the character from an image
def predict_character(img_path):
    try:
        processed_image = preprocess_image(img_path)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability
        predicted_label = label_mapping[predicted_class]  # Map index back to character
        return predicted_label
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")

# Example usage: Test with a list of image paths
test_images = [
    'test_k.jpg',  # Update with actual paths to your test images
    'test_d.png',
    'test_f.png',
    'test_g.jpg',
    'test_l.png',
    'test_z.png',
    'test_n.png',
    # Add more image paths as needed
]

for img_path in test_images:
    predicted_label = predict_character(img_path)
    print(f'Predicted label for {img_path}: {predicted_label}')
