import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================================
# 🎨 UI CONFIGURATION
# =========================================
st.set_page_config(
    page_title="Speech Emotion Detection",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .emotion-header {
        font-size: 30px;
        font-weight: bold;
        text-align: center;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================
# 🧠 MODEL LOADING
# =========================================

@st.cache_resource
def load_all_models():
    """Loads the model, scaler, and label encoder once to avoid reloading on every run."""
    try:
        model = load_model("final_emotion_model.keras")
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("label_encoder.pkl")
        return model, scaler, le
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

model, scaler, le = load_all_models()

# =========================================
# 🔢 FEATURE EXTRACTION
# =========================================

def extract_features(file_path):
    """Extracts MFCC features from the audio file."""
    try:
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# =========================================
# 🔮 PREDICTION FUNCTION
# =========================================

def predict_emotion(audio_path):
    features = extract_features(audio_path)
    if features is None:
        return None, None

    # Scale features
    features = scaler.transform([features])

    # Predict
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    predicted_label = le.inverse_transform([predicted_index])[0]
    
    # Get probabilities
    class_probs = prediction[0]
    classes = le.classes_
    confidence_scores = {classes[i]: float(class_probs[i]) for i in range(len(classes))}
    
    return predicted_label, confidence_scores

# =========================================
# 🖥️ SIDEBAR
# =========================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3048/3048122.png", width=100)
    st.title("Project Info")
    st.info("This application uses a Deep Learning model (CNN/Dense) to detect emotions from speech audio.")
    
    st.subheader("Model Info")
    st.text("Type: Neural Network (Keras)")
    st.text("Features: MFCC")
    st.text("Input: WAV Audio")
    
    st.subheader("Supported Emotions")
    if le:
        st.write(", ".join(le.classes_))
    
    st.markdown("---")
    st.caption("Developed with ❤️ using Streamlit")

# =========================================
# 🏠 MAIN INTERFACE
# =========================================

st.title("🎙️ Speech Emotion Detection System")
st.markdown("Upload an audio file to analyze the speaker's emotion.")

# 1. File Uploader
uploaded_file = st.file_uploader("Upload Audio File (.wav)", type=["wav"])

if uploaded_file is not None:
    # Save the file temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display Audio Player
    st.audio(temp_path, format="audio/wav")
    
    # 2. Prediction Button
    if st.button("🔍 Analyze Emotion"):
        with st.spinner("Analyzing..."):
            emotion, scores = predict_emotion(temp_path)
            
            if emotion:
                # 3. Display Results
                st.markdown("---")
                
                # Emoji mapping
                emoji_dict = {
                    "neutral": "😐", "calm": "😌", "happy": "😊", "sad": "😢",
                    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲"
                }
                
                main_emoji = emoji_dict.get(emotion, "🎤")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"<div class='emotion-header'>{main_emoji}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='emotion-header'>{emotion.upper()}</div>", unsafe_allow_html=True)
                    conf = scores[emotion]
                    st.metric("Confidence", f"{conf:.2%}")
                
                with col2:
                    st.subheader("📊 Confidence Scores")
                    # Create DataFrame for chart
                    df_scores = pd.DataFrame(list(scores.items()), columns=["Emotion", "Score"])
                    df_scores = df_scores.sort_values(by="Score", ascending=False)
                    
                    st.bar_chart(df_scores.set_index("Emotion"))
                
                # Success Message
                st.success(f"Prediction Complete! The speaker sounds **{emotion}**.")
                
            else:
                st.error("Failed to process audio.")
            
    # Cleanup (Optional)
    # os.remove(temp_path) 
