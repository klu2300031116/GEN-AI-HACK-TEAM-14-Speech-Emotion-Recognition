import librosa
import numpy as np
import joblib
import os
print("Loading libraries and model...")

from tensorflow.keras.models import load_model

# =========================
# Load Saved Components
# =========================

model = load_model("final_emotion_model.keras")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# =========================
# Feature Extraction
# =========================

def extract_features(file_path = r"C:\Users\hrish\Downloads\Audio_Song_Actors_01-24\Actor_04"):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# =========================
# Select Audio File
# =========================

import os
import random

# =========================
# Select Audio File
# =========================

# Directory containing the dataset
DATASET_DIR = "dataset"

# Function to get a random wav file
def get_random_wav(dataset_dir):
    wav_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    
    if not wav_files:
        return None
    return random.choice(wav_files)

file_path = get_random_wav(DATASET_DIR)

if file_path:
    print(f"Selected Audio File: {file_path}")
else:
    print("No .wav files found in the dataset directory.")
    exit()

# 👆 Now selects a random file for testing

# =========================
# Process File
# =========================

features = extract_features(file_path)
features = scaler.transform([features])

# =========================
# Predict
# =========================

prediction = model.predict(features)
predicted_label = le.inverse_transform([np.argmax(prediction)])

print("Predicted Emotion:", predicted_label[0])
