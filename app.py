import streamlit as st
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import tempfile
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import soundfile as sf

# =========================================
# 🎨 PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Speech Emotion Detection",
    page_icon="🎙️",
    layout="centered"
)

# =========================================
# 🧠 LOAD MODEL
# =========================================
@st.cache_resource
def load_all_models():
    model = load_model("final_emotion_model.keras")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, scaler, le

model, scaler, le = load_all_models()

# =========================================
# 📊 FEATURE EXTRACTION
# =========================================
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# =========================================
# 🔮 PREDICT FUNCTION
# =========================================
def predict_emotion(audio_path):
    features = extract_features(audio_path)
    features = scaler.transform([features])
    prediction = model.predict(features)

    predicted_index = np.argmax(prediction)
    predicted_label = le.inverse_transform([predicted_index])[0]

    class_probs = prediction[0]
    classes = le.classes_

    scores = {
        classes[i]: float(class_probs[i])
        for i in range(len(classes))
    }

    return predicted_label, scores

# =========================================
# 📌 SIDEBAR
# =========================================
with st.sidebar:
    st.title("📌 Project Info")
    st.info("Deep Learning based Speech Emotion Detection System")

    st.subheader("Model Info")
    st.write("Type: CNN + Dense")
    st.write("Features: MFCC")
    st.write("Input: WAV / MP3")

    st.subheader("Supported Emotions")
    st.write(", ".join(le.classes_))

    st.subheader("Model Accuracy")
    st.progress(0.86)  # change if needed
    st.write("Accuracy: 86%")

    st.markdown("---")
    st.caption("Developed by Hrishitha ❤️")

# =========================================
# 🏠 MAIN TITLE
# =========================================
st.title("🎙️ Speech Emotion Detection System")
st.markdown("Upload WAV/MP3 file or use Live Microphone.")

# =========================================
# 📂 FILE UPLOAD
# =========================================
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.audio(temp_path)

    if st.button("🔍 Analyze Uploaded Audio"):

        emotion, scores = predict_emotion(temp_path)

        st.markdown("---")

        col1, col2 = st.columns([1,2])

        with col1:
            st.subheader("Predicted Emotion")
            st.success(emotion.upper())
            st.metric("Confidence", f"{scores[emotion]*100:.2f}%")

        with col2:
            df = pd.DataFrame(
                scores.items(),
                columns=["Emotion", "Confidence"]
            ).sort_values(by="Confidence", ascending=False)

            st.bar_chart(df.set_index("Emotion"))

# =========================================
# 🎤 LIVE MICROPHONE
# =========================================
st.markdown("---")
st.subheader("🎤 Live Microphone Recognition")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

if webrtc_ctx.audio_processor:

    if st.button("Analyze Live Recording"):

        audio_frames = webrtc_ctx.audio_processor.frames

        if len(audio_frames) > 0:

            audio_data = np.concatenate(audio_frames, axis=1)

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(temp_file.name, audio_data.T, 44100)

            emotion, scores = predict_emotion(temp_file.name)

            st.markdown("---")

            col1, col2 = st.columns([1,2])

            with col1:
                st.subheader("Live Emotion")
                st.success(emotion.upper())
                st.metric("Confidence", f"{scores[emotion]*100:.2f}%")

            with col2:
                df = pd.DataFrame(
                    scores.items(),
                    columns=["Emotion", "Confidence"]
                ).sort_values(by="Confidence", ascending=False)

                st.bar_chart(df.set_index("Emotion"))
