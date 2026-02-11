import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical

# =========================
# 1️⃣ DATASET PATH
# =========================

DATA_PATH = "dataset"

# Emotion dictionary for RAVDESS
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised"
}

# =========================
# 2️⃣ LOAD FILE PATHS
# =========================

file_paths = []
labels = []

for actor in os.listdir(DATA_PATH):
    actor_path = os.path.join(DATA_PATH, actor)

    if os.path.isdir(actor_path):

        for file in os.listdir(actor_path):
            if file.endswith(".wav"):

                file_paths.append(os.path.join(actor_path, file))

                # Extract emotion from filename (RAVDESS uses "-")
                parts = file.split("-")

                if len(parts) > 2:
                    emotion_code = parts[2]
                    emotion = emotion_dict.get(emotion_code)
                    labels.append(emotion)

print("Total files loaded:", len(file_paths))

# =========================
# 3️⃣ FEATURE EXTRACTION
# =========================

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

features = []

for file in file_paths:
    features.append(extract_features(file))

X = np.array(features)

# =========================
# 4️⃣ ENCODE LABELS
# =========================

le = LabelEncoder()
y = le.fit_transform(labels)
y = to_categorical(y)

# =========================
# 5️⃣ TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Reshape for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# =========================
# 6️⃣ BUILD CNN MODEL
# =========================

model = Sequential()

model.add(Conv1D(256, 5, padding='same',
                 input_shape=(X_train.shape[1], 1),
                 activation='relu'))
model.add(MaxPooling1D(pool_size=5))

model.add(Conv1D(128, 5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(y.shape[1], activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# =========================
# 7️⃣ TRAIN MODEL
# =========================

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# =========================
# 8️⃣ EVALUATE MODEL
# =========================

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# =========================
# 9️⃣ SAVE MODEL
# =========================

model.save("emotion_model.h5")

print("Model saved successfully!")
