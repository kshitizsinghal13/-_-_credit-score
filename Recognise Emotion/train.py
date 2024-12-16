#pip install -r requirements.txt
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


DATASET_PATH = 'main'  
MODEL_PATH = 'emotion_model.h5'  





# Function to extract features from audio files
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean, sample_rate 





# Function to augment audio data
def augment_audio(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    
    # Randomly choose an augmentation
    if np.random.rand() < 0.5:
        audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))  # Time stretching
    if np.random.rand() < 0.5:
        audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=np.random.randint(-2, 3))  # Pitch shifting
    if np.random.rand() < 0.5:
        noise = np.random.randn(len(audio)) * 0.005  # Adding noise
        audio = audio + noise
    
    return audio, sample_rate  # Return augmented audio and its sample rate







# Load dataset and extract features with augmentation
def load_data():
    features = []
    labels = []
    
    for file in os.listdir(DATASET_PATH):
        if file.endswith('.wav'):
            emotion_label = file.split('-')[2]  
            emotion_mapping = {
                '03': 'happy',
                '04': 'sad',
                '05': 'angry',
                '06': 'fearful',
                '07': 'disgust',
                '08': 'surprised',
                '01': 'neutral',
                '02': 'calm'
            }
            emotion = emotion_mapping.get(emotion_label)
            if not emotion:
                continue
            
            file_path = os.path.join(DATASET_PATH, file)
            mfccs_mean, sample_rate = extract_features(file_path)  # Get both MFCCs and sample rate
            features.append(mfccs_mean)
            labels.append(emotion)

            augmented_audio, _ = augment_audio(file_path)  # Get augmented audio and ignore sample rate

            # Ensure augmented audio is valid before processing
            if len(augmented_audio) > 0:
                # Use the original sample rate for augmented audio MFCC calculation
                mfccs_mean_augmented = np.mean(librosa.feature.mfcc(y=augmented_audio, sr=sample_rate, n_mfcc=40).T, axis=0)
                features.append(mfccs_mean_augmented)
                labels.append(emotion)

    return np.array(features), np.array(labels)





# Function to train the model
def train_model():
    X, y = load_data()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    y_categorical = to_categorical(y_encoded)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.1, random_state=42)

    # Reshape data for CNN (adding a channel dimension)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)  
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

    # Build CNN model 
    model = Sequential()
    model.add(Conv2D(32, (3, 1), activation='relu', input_shape=(X_train.shape[1], 1, 1)))  
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))  

    model.add(Conv2D(64, (3, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))  
    model.add(Dense(len(np.unique(y)), activation='softmax'))
    
    
    
    

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define Early Stopping and Learning Rate Reduction callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    # Train the model with Early Stopping and Learning Rate Reduction
    model.fit(X_train, y_train,
              epochs=100,
              batch_size=32,
              validation_data=(X_test, y_test),
              callbacks=[early_stopping, reduce_lr])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    
    
    model.save(MODEL_PATH)
    
    
    np.save('label_encoder.npy', label_encoder.classes_) 
    
   
    return model, label_encoder  

if __name__ == "__main__":
    
   # Train the model and get it ready for predictions.
   trained_model, label_encoder = train_model()  # Get both trained model and label encoder
