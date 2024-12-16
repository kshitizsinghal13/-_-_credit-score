#pip install -r requirements.txt
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Constants
MODEL_PATH = 'emotion_model.h5'  # Path where the trained model is saved






# Function to extract features from audio files
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean  # Return MFCCs only

# Load saved label encoder






def load_label_encoder():
   return np.load('label_encoder.npy', allow_pickle=True)





# Function to predict emotion from a single audio file using a loaded model
def predict_emotion(model):
   file_path = input("Enter the path of the audio file: ")
   
   if not os.path.exists(file_path):
       print("File does not exist.")
       return
   
   mfccs_mean = extract_features(file_path)  
   mfccs_mean = mfccs_mean.reshape(1, -1)  
   mfccs_mean = mfccs_mean.reshape(mfccs_mean.shape[0], mfccs_mean.shape[1], 1) 

   prediction = model.predict(mfccs_mean)
   
   predicted_class_index = np.argmax(prediction)

   emotion_label = label_encoder.inverse_transform([predicted_class_index])[0]
   
   print(f"Predicted Emotion: {emotion_label}")







if __name__ == "__main__":
   # Load the trained model from disk.
   trained_model = load_model(MODEL_PATH)

   label_encoder_classes = load_label_encoder()
   label_encoder = LabelEncoder()
   label_encoder.classes_ = label_encoder_classes

   predict_emotion(trained_model) 