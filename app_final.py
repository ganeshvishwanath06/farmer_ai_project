import os
import tempfile
import urllib.request
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import openai
import requests
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment

# ------------------------
# FFmpeg setup for Windows
# ------------------------
ffmpeg_bin = r"C:\Users\aman2\OneDrive\Desktop\ffmpeg\ffmpeg-8.0-essentials_build\bin"
ffmpeg_path = os.path.join(ffmpeg_bin, "ffmpeg.exe")
ffprobe_path = os.path.join(ffmpeg_bin, "ffprobe.exe")

if not os.path.isfile(ffmpeg_path) or not os.path.isfile(ffprobe_path):
    raise FileNotFoundError(f"FFmpeg or FFprobe not found:\n{ffmpeg_path}\n{ffprobe_path}")

os.environ["FFMPEG_BINARY"] = ffmpeg_path
os.environ["FFPROBE_BINARY"] = ffprobe_path

AudioSegment.converter = ffmpeg_path
AudioSegment.ffmpeg = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

# ------------------------
# OpenAI API key
# ------------------------
openai.api_key = "YOUR_OPENAI_KEY_HERE"  # 🔑 Replace with your key

# ------------------------
# Disease classes
# ------------------------
disease_classes = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_","Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy","Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy","Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew","Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus","Tomato___healthy"
]

# ------------------------
# Image transform
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------------
# Load disease model
# ------------------------
MODEL_PATH = "plant_disease_model.pth"
MODEL_URL = "https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification/resolve/main/pytorch_model.bin"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info(f"Downloading model from {MODEL_URL}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model download complete!")

    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 38)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

model = load_model()

# ------------------------
# GPT helper
# ------------------------
def generate_advice(query, language):
    lang_map = {
        "English": "Reply in simple English for farmers.",
        "Hindi": "किसानों के लिए सरल हिंदी में जवाब दें।",
        "Malayalam": "കർഷകർക്ക് എളുപ്പത്തിൽ മനസ്സിലാകുന്ന മലയാളത്തിൽ മറുപടി നൽകുക."
    }
    prompt = f"Farmer's query: {query}\n{lang_map[language]}"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are an agriculture expert."},
            {"role":"user","content":prompt}
        ]
    )
    return response.choices[0].message["content"]

# ------------------------
# Text to Speech helper
# ------------------------
def speak_text(text, language):
    lang_codes = {"English":"en","Hindi":"hi","Malayalam":"ml"}
    tts = gTTS(text=text, lang=lang_codes[language])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts.save(temp_audio.name)
        return temp_audio.name

# ------------------------
# Pages
# ------------------------
def page_home():
    st.title("🌾 Digital Krishi Officer")
    st.write("Welcome! Choose a service from the sidebar.")

def page_photo_diagnosis():
    st.title("📷 Crop Disease Diagnosis")
    language = st.radio("Select language:", ["English","Hindi","Malayalam"])
    uploaded_file = st.file_uploader("Upload a crop leaf photo", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf", use_column_width=True)

        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs,1)
            disease = disease_classes[predicted.item()]

        st.success(f"✅ Detected: {disease}")
        advice = generate_advice(f"Disease detected: {disease}", language)
        st.info(f"💡 Advice: {advice}")

        audio_path = speak_text(advice, language)
        st.audio(audio_path)

        feedback = st.radio("Did this advice help?", ["Yes","No"])
        if feedback=="No":
            st.warning("📨 This case will be flagged for expert review.")

def page_weather():
    st.title("☀️ Weather Info")
    city = st.text_input("Enter your city:")
    if city:
        try:
            url = f"http://wttr.in/{city}?format=%C+%t"
            weather = requests.get(url).text.strip()
            st.success(f"🌤 Weather in {city}: {weather}")
        except:
            st.error("❌ Could not fetch weather data.")

def page_schemes():
    st.title("📑 Government Schemes")
    st.write("""
    ✅ PM-Kisan Samman Nidhi  
    ✅ Soil Health Card  
    ✅ Pradhan Mantri Fasal Bima Yojana  
    ✅ Kisan Credit Card (KCC)  
    """)

def page_ask_question():
    st.title("🎤 Ask a Question")
    language = st.radio("Select language:", ["English","Hindi","Malayalam"])
    audio_bytes = st.audio_input("🎙️ Record your question")
    query = st.text_area("✍️ Or type your question here:")

    if audio_bytes is not None:
        st.success("✅ Audio recorded, processing...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(audio_bytes.getbuffer())
            temp_audio_path = temp_audio.name

        # Convert WebM → WAV
        wav_path = temp_audio_path.replace(".webm", ".wav")
        try:
            sound = AudioSegment.from_file(temp_audio_path, format="webm")
            sound.export(wav_path, format="wav")
        except Exception as e:
            st.error(f"❌ Audio conversion failed: {e}")
            return

        # Speech recognition
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(wav_path) as source:
                audio = recognizer.record(source)
                query = recognizer.recognize_google(
                    audio,
                    language="hi" if language=="Hindi" else "ml" if language=="Malayalam" else "en"
                )
                st.success(f"🗣 Recognized speech: {query}")
        except sr.UnknownValueError:
            st.error("❌ Could not understand audio.")
        except sr.RequestError:
            st.error("⚠️ Speech recognition service unavailable.")
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if os.path.exists(wav_path):
                os.remove(wav_path)

    if st.button("Get Answer") and query:
        answer = generate_advice(query, language)
        st.info(f"💡 Answer: {answer}")
        audio_path = speak_text(answer, language)
        st.audio(audio_path)

# ------------------------
# Navigation
# ------------------------
PAGES = {
    "🏠 Home": page_home,
    "📷 Photo Diagnosis": page_photo_diagnosis,
    "☀️ Weather": page_weather,
    "📑 Schemes": page_schemes,
    "🎤 Ask a Question": page_ask_question,
}

st.sidebar.title("🔎 Navigate")
choice = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[choice]()
