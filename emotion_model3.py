import os
import joblib

import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# =====================
# DATASET PATH
# =====================

DATA_PATH = "dataset"

file_paths = []
labels = []

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# =====================
# LOAD FILES
# =====================

for actor in os.listdir(DATA_PATH):
    actor_path = os.path.join(DATA_PATH, actor)

    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            parts = file.split("-")
            emotion_code = parts[2]

            if emotion_code in emotion_dict:
                file_paths.append(os.path.join(actor_path, file))
                labels.append(emotion_dict[emotion_code])

print("Total files:", len(file_paths))

# =====================
# FEATURE EXTRACTION (MFCC only)
# =====================

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

features = []

for file in file_paths:
    try:
        features.append(extract_features(file))
    except:
        continue

X = np.array(features)
y = np.array(labels[:len(X)])

# =====================
# SCALE
# =====================

scaler = StandardScaler()
X = scaler.fit_transform(X)

# =====================
# ENCODE LABELS
# =====================

le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

# =====================
# SPLIT (IMPORTANT)
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # 🔥 prevents imbalance
)

# =====================
# SIMPLE DENSE MODEL
# =====================

model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# =====================
# TRAIN
# =====================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# =====================
# EVALUATE
# =====================

loss, accuracy = model.evaluate(X_test, y_test)
print("Final Test Accuracy:", accuracy)

model.save("final_emotion_model.keras")
# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save label encoder
joblib.dump(le, "label_encoder.pkl")

print("Model, Scaler, and Label Encoder saved successfully!")


print("Model saved successfully!")
