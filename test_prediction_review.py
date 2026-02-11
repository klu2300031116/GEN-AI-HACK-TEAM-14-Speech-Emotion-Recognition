import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# =========================
# Load Saved Components
# =========================

model_path = "final_emotion_model.keras"
scaler_path = "scaler.pkl"
le_path = "label_encoder.pkl"

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found.")
    exit()

model = load_model(model_path)
scaler = joblib.load(scaler_path)
le = joblib.load(le_path)

# =========================
# Feature Extraction
# =========================

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# =========================
# Select Audio File
# =========================

file_path = "dataset/Actor_24/03-02-06-02-02-01-24.wav"

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
    exit()

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

print(f"Test File: {file_path}")
print(f"Predicted Emotion: {predicted_label[0]}")
