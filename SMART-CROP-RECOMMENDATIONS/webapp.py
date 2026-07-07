## Importing necessary libraries for the web app
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from PIL import Image
import warnings
from sklearn.model_selection import train_test_split
from gtts import gTTS
from io import BytesIO
import requests # To fetch images from the web

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌿",
    layout="wide"
)

# --- Model Management ---
@st.cache_resource
def load_model():
    try:
        with open('RF.pkl', 'rb') as model_file:
            return pickle.load(model_file)
    except FileNotFoundError:
        return None

model = load_model()

# --- Prediction Function ---
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    if model:
        prediction = model.predict(np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]))
        return prediction
    return None

# --- Function to fetch an image from Unsplash ---
def get_crop_image_url(crop_name):
    # This is a simple way to get a relevant image. For more reliability, you might need an API key.
    return f"https://source.unsplash.com/800x600/?{crop_name.lower()}"

# --- Main App Interface ---
def main():
    try:
        img = Image.open("crop.png")
        st.image(img, use_column_width=True)
    except FileNotFoundError:
        st.warning("Header image 'crop.png' not found.")

    st.markdown("<h1 style='text-align: center;'>SMART CROP RECOMMENDATION</h1>", unsafe_allow_html=True)
    
    st.sidebar.title("AgriSense AI")
    st.sidebar.header("Enter Crop Details")
    nitrogen = st.sidebar.slider("Nitrogen (N)", 0, 140, 40)
    phosphorus = st.sidebar.slider("Phosphorus (P)", 5, 145, 50)
    potassium = st.sidebar.slider("Potassium (K)", 5, 205, 50)
    temperature = st.sidebar.slider("Temperature (°C)", 9.0, 44.0, 25.0)
    humidity = st.sidebar.slider("Humidity (%)", 14.0, 100.0, 70.0)
    ph = st.sidebar.slider("Soil pH Level", 3.5, 9.9, 6.5)
    rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 299.0, 100.0)
    
    if st.sidebar.button("Predict Crop Recommendation"):
        if model is None:
            st.error("Model file (RF.pkl) not found. Please ensure 'Crop_recommendation.csv' is present to train the model on first run.")
        else:
            prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            
            if prediction:
                crop_name = prediction[0]
                
                st.success(f"**Recommended Crop is: {crop_name}**")
                
                try:
                    speech_text = f"The recommended crop is {crop_name}"
                    tts = gTTS(text=speech_text, lang='en', slow=False)
                    audio_fp = BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)
                    st.audio(audio_fp, format='audio/mp3')
                except Exception as e:
                    st.error(f"Could not generate voice output. Check internet connection. Error: {e}")
                
                # Fetch and display image from online source
                image_url = get_crop_image_url(crop_name)
                st.image(image_url, caption=f"An image of {crop_name}", use_column_width=True)

if __name__ == '__main__':
    main()