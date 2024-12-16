import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Set parameters
img_height, img_width = 28, 28
batch_size = 32

# Load dataset from CSV file
csv_file_path = 'english.csv'  # Update this path to your CSV file location
data = pd.read_csv(csv_file_path)

# Prepare image data and labels
images = []
labels = []

# Create a mapping from characters to integers
label_mapping = {char: idx for idx, char in enumerate(sorted(data['label'].unique()))}

for index, row in data.iterrows():
    img_path = row['image']
    label = row['label']
    
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        # Resize image to 28x28 pixels
        img = cv2.resize(img, (img_width, img_height))
        images.append(img)
        labels.append(label_mapping[label])  # Map label to integer using the mapping
    else:
        print(f"Warning: Unable to load image at {img_path}. Check the file path.")

# Convert lists to numpy arrays and normalize pixel values
images = np.array(images).reshape(-1, img_height, img_width, 1) / 255.0
labels = np.array(labels)

# Convert labels to categorical format
num_classes = len(label_mapping)
labels = to_categorical(labels, num_classes)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data Augmentation with additional techniques
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    brightness_range=[0.7, 1.3],
    shear_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest')

datagen.fit(X_train)

# Define a more complex CNN model with more layers and neurons
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    layers.BatchNormalization(),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    
    layers.Flatten(),
    
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    
    layers.Dense(num_classes, activation='softmax')
])

# Compile model with a lower learning rate if necessary
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for model checkpointing and learning rate reduction
model_checkpoint = ModelCheckpoint('best_handwritten_model.keras', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train model using data augmentation generator with increased epochs
model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
          epochs=100,
          validation_data=(X_test, y_test),
          callbacks=[model_checkpoint, reduce_lr])  # Removed early stopping

# Evaluate the model on test data using the best saved model
model.load_weights('best_handwritten_model.keras')  # Load the best weights from training

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Save the final model after training in .keras format if needed (optional)
model.save('handwritten_character_recognition_model.keras')
