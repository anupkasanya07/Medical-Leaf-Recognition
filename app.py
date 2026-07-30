import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import pandas as pd
import google.generativeai as genai
import base64

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyDPiyjFYqXJrXfes3HMOkC-6TnMdEt5Igw"
genai.configure(api_key=GEMINI_API_KEY)

# Load Model
MODEL_PATH = r"D:\COLLEGE\SEMESTER-5\PROJECT\trained data\final_data.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load CSV Data
df = pd.read_csv(r'D:\COLLEGE\SEMESTER-5\PROJECT\re-proj\Final_proj\plant_classification.csv')

# Function to extract plant information
def infoextract(indexnum):
    return {
        "Botanical Name": df['Botanical Name'][indexnum],
        "Common Name": df['Common Name'][indexnum],
        "Family": df['Family'][indexnum],
        "Bioactive Compounds": df['Bioactive Compounds'][indexnum],
        "Traditional Uses": df['Traditional Uses'][indexnum],
    }

# Image Preprocessing
def preprocess_image(image):
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Image Classification
def classify(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    return infoextract(predicted_class_index)

# Gemini AI Response
def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text if response else "Sorry, I couldn't process your request."

# Streamlit Chat UI
st.set_page_config(page_title="ChatBot & Plant Classifier", layout="wide")
st.title("🤖 ChatBot & 🌱 Plant Classifier")

# Session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_input = st.chat_input("Ask me anything or upload an image for classification:")

# Handle User Query
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = get_gemini_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Image Upload for Classification
uploaded_image = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    
    # Convert image to bytes for inline display
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format="PNG")
    encoded_image = base64.b64encode(img_byte_array.getvalue()).decode()
    
    # Display small image in chat area
    with st.chat_message("user"):
        st.markdown(f'<img src="data:image/png;base64,{encoded_image}" width="150" height="150">', unsafe_allow_html=True)
    
    # Perform classification
    plant_info = classify(image)
    gemini_prompt = f"Can you provide additional details about this plant? Here is the data: {plant_info}"
    ai_response = get_gemini_response(gemini_prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(f"### 🌱 Plant Classification\n\n{ai_response}")