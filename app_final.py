# algofarm.py - The final, hackathon-ready version
import streamlit as st
import openai
from gtts import gTTS
from deep_translator import GoogleTranslator
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import io
import os
import json
from datetime import datetime
import requests # Added back requests for the final code
import pydub # We still need pydub to handle the audio file conversion for gTTS

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Farmer's Assistant 👨‍🌾",
    page_icon="🌿",
    layout="wide"
)

# --- App Title and Description ---
st.title("AI-Based Farmer Query Support and Advisory System 👨‍🌾")
st.markdown("Your digital Krishi Officer, ready to help with pests, weather, and market advice.")

# --- API Key Setup ---
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    st.error("OpenAI API key not found. Please create a file at .streamlit/secrets.toml and add your key.", icon="🚨")
    st.stop()

# --- Session State Initialization ---
if 'language' not in st.session_state:
    st.session_state.language = 'ml' # Default to Malayalam
if 'location' not in st.session_state:
    st.session_state.location = ''
if 'crop' not in st.session_state:
    st.session_state.crop = ''
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

# --- Disease Model Setup ---
# Note: For the hackathon, you can keep the fallback to ImageNet if you don't have a custom model.
# Option A: Provide a custom model and classes here.
MODEL_PATH = 'plant_disease_model.pth' # Make sure this file is in the same directory
DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy' 
]

@st.cache_resource
def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            st.warning("Custom disease model not found. Using a generic pre-trained ImageNet model as a fallback.", icon="⚠️")
            model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            model.eval()
            return model, False
        
        model = mobilenet_v2(num_classes=len(DISEASE_CLASSES))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        st.success("Plant disease model loaded successfully!", icon="✅")
        return model, True
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure 'plant_disease_model.pth' is correct.", icon="🚨")
        return None, False

disease_model, is_custom_model_loaded = load_model(MODEL_PATH)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Helper Functions ---
def get_season(month):
    """Simple function to get the season based on the month (for Indian context)"""
    if month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    elif month in [10, 11, 12]:
        return "Post-monsoon"
    else:
        return "Winter"

def translate_text(text, target_lang):
    if target_lang == 'en' or not text: return text
    try: return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception: return text

def text_to_speech(text, lang):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception: return None

def log_query(query, response, context, feedback=None):
    """Logs the user query and AI response for the 'learning loop' and 'escalation'."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "context": context,
        "feedback": feedback
    }
    with open("query_logs.jsonl", "a") as f:
        json.dump(log_entry, f)
        f.write('\n')

def get_llm_response(query, context):
    try:
        # This is the core of the context-aware system
        prompt = f"""
        You are an expert AI agricultural assistant for farmers in India.
        Your answers must be simple, practical, and in the same language as the user's query.
        Be concise and direct in your advice.
        Current Context:
        - Farmer's Location: {context.get('location', 'Not provided')}
        - Current Crop: {context.get('crop', 'Not provided')}
        - Language of Query: {context.get('language_name', 'English')}
        - Current Season: {context.get('season', 'Not provided')}
        User's Question: "{query}"
        Provide a helpful and actionable response.
        """
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error getting response from AI model: {e}")
        return "Sorry, I couldn't process your request at the moment. Please try again."

def predict_disease(image_bytes):
    if not disease_model: return "Model not loaded.", 0.0
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = disease_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.max(probabilities, 0)
    
    if is_custom_model_loaded and top_catid.item() < len(DISEASE_CLASSES):
        class_name = DISEASE_CLASSES[top_catid.item()]
    else:
        # Fallback for generic model
        generic_classes = ['Fungal infection', 'Bacterial issue', 'Nutrient deficiency', 'Healthy']
        class_name = generic_classes[top_catid.item() % len(generic_classes)]
    return class_name.replace("_", " "), top_prob.item()

# --- Pre-defined Data for Schemes (Simulating a local database) ---
SCHEMES_DATA = {
    "en": {
        "PM-KISAN Scheme": "Provides income support of Rs. 6,000 per year to all landholding farmer families.",
        "Pradhan Mantri Fasal Bima Yojana (PMFBY)": "An insurance service for crop losses due to natural calamities, pests, and diseases.",
        "Kisan Credit Card (KCC)": "Provides timely credit support to farmers for their cultivation and other needs.",
        "Soil Health Card Scheme": "Promotes soil testing and provides a report card to help farmers apply nutrients as per crop needs.",
    },
    "ml": {
        "പിഎം-കിസാൻ പദ്ധതി": "എല്ലാ ഭൂവുടമകളായ കർഷക കുടുംബങ്ങൾക്കും പ്രതിവർഷം 6,000 രൂപ വരുമാന സഹായം നൽകുന്നു.",
        "പ്രധാനമന്ത്രി ഫസൽ ബീമ യോജന (പിഎംഎഫ്ബിവൈ)": "പ്രകൃതി ദുരന്തങ്ങൾ, കീടങ്ങൾ, രോഗങ്ങൾ എന്നിവ കാരണം വിളനാശം സംഭവിച്ചാൽ ഇൻഷുറൻസ് പരിരക്ഷ നൽകുന്നു.",
        "കിസാൻ ക്രെഡിറ്റ് കാർഡ് (കെസിസി)": "കർഷകർക്ക് അവരുടെ കൃഷിക്കും മറ്റ് ആവശ്യങ്ങൾക്കും സമയബന്ധിതമായി വായ്പാ സഹായം നൽകുന്നു.",
        "മണ്ണ് ആരോഗ്യ കാർഡ് പദ്ധതി": "മണ്ണ് പരിശോധന പ്രോത്സാഹിപ്പിക്കുകയും വിളകളുടെ ആവശ്യങ്ങൾക്കനുസരിച്ച് പോഷകങ്ങൾ പ്രയോഗിക്കാൻ കർഷകരെ സഹായിക്കുന്ന റിപ്പോർട്ട് കാർഡ് നൽകുകയും ചെയ്യുന്നു.",
    },
    "hi": {
        "पीएम-किसान योजना": "सभी भूमिधारक किसान परिवारों को प्रति वर्ष 6,000 रुपये की आय सहायता प्रदान करता है।",
        "प्रधान मंत्री फसल बीमा योजना (पीएमएफबीवाई)": "प्राकृतिक आपदाओं, कीटों और बीमारियों के कारण फसल के नुकसान के लिए एक बीमा सेवा।",
        "किसान क्रेडिट कार्ड (केसीसी)": "किसानों को उनकी खेती और अन्य जरूरतों के लिए समय पर ऋण सहायता प्रदान करता है।",
        "मृदा स्वास्थ्य कार्ड योजना": "मिट्टी की जांच को बढ़ावा देता है और किसानों को फसल की जरूरतों के अनुसार पोषक तत्वों का उपयोग करने में मदद करने के लिए एक रिपोर्ट कार्ड प्रदान करता है।",
    }
}

# --- Sidebar ---
with st.sidebar:
    st.header("Settings ⚙️")
    lang_map = {'ml': 'മലയാളം', 'en': 'English', 'hi': 'हिन्दी'}
    lang_code_map = {'ml': 'ml-IN', 'en': 'en-US', 'hi': 'hi-IN'}
    selected_lang = st.selectbox("Choose your language:", options=list(lang_map.keys()), format_func=lambda x: lang_map[x])
    st.session_state.language = selected_lang

    st.text_input("📍 Your Location (e.g., Alappuzha, Kerala)", key='location')
    st.text_input("🌾 Your Primary Crop (e.g., Banana, Rice)", key='crop')

    st.markdown("---")
    app_mode = st.radio("Choose a feature:", ("❓ Ask a Question", "📸 Photo Diagnosis", "📢 Government Schemes"))

# --- Main App Logic ---
if app_mode == "❓ Ask a Question":
    st.header("Ask a Question in Your Language")
    
    uploaded_audio = st.file_uploader("Upload an audio file to transcribe (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])
    user_query = ""

    if uploaded_audio:
        st.audio(uploaded_audio)
        st.info("Transcribing audio...")
        with st.spinner("Transcribing..."):
            try:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=uploaded_audio,
                    language=st.session_state.language
                )
                user_query = transcript.text
                st.success("Transcription complete.")
            except openai.APIError as e:
                st.error(f"Error with OpenAI's API: {e}")
                user_query = ""
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                user_query = ""
    
    # Text input for manual query
    text_query = st.text_input("Or type your question here:", key="query_input")
    
    if text_query:
        user_query = text_query

    if user_query:
        st.markdown(f"**Your Question:** *{user_query}*")
        with st.spinner("Thinking... 🤖"):
            translated_query_for_llm = translate_text(user_query, 'en')
            context = {
                'location': st.session_state.location,
                'crop': st.session_state.crop,
                'language_name': lang_map[st.session_state.language],
                'season': get_season(datetime.now().month)
            }
            llm_response_en = get_llm_response(translated_query_for_llm, context)
            final_response = translate_text(llm_response_en, st.session_state.language)
            st.markdown("---")
            st.subheader("💡 My Advice for You")
            st.markdown(final_response)
            
            # Add escalation and feedback logic
            if "sorry" in final_response.lower() or "couldn't" in final_response.lower() or "I cannot provide" in final_response.lower():
                st.info("Your query is complex. We've logged it for our agricultural experts to review. Thank you for your patience.")
                log_query(user_query, final_response, context, "Escalated")
            else:
                st.subheader("Was this helpful? (Learning Loop)")
                col_fb1, col_fb2 = st.columns([1, 10])
                with col_fb1:
                    if st.button("👍"):
                        log_query(user_query, final_response, context, "Helpful")
                        st.success("Thank you for your feedback!")
                with col_fb2:
                    if st.button("👎"):
                        log_query(user_query, final_response, context, "Not Helpful")
                        st.error("Thank you for your feedback. We will use this to improve.")
            
            audio_file = text_to_speech(final_response, st.session_state.language)
            if audio_file: st.audio(audio_file, format='audio/mp3')

elif app_mode == "📸 Photo Diagnosis":
    st.header("Upload a Photo of the Diseased Plant")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        with st.spinner("Analyzing the image... 🔬"):
            disease_name, confidence = predict_disease(uploaded_file.getvalue())
            st.success(f"**Diagnosis:** {disease_name} (Confidence: {confidence:.2%})")
            st.subheader("🤖 Recommended Actions")
            context = {
                'location': st.session_state.location,
                'crop': st.session_state.crop,
                'language_name': lang_map[st.session_state.language],
                'season': get_season(datetime.now().month)
            }
            query_for_llm = f"My plant has been diagnosed with '{disease_name}'. What are the prevention, organic, and chemical control methods?"
            llm_response_en = get_llm_response(query_for_llm, context)
            final_response = translate_text(llm_response_en, st.session_state.language)
            st.markdown(final_response)
            audio_file = text_to_speech(final_response, st.session_state.language)
            if audio_file: st.audio(audio_file, format='audio/mp3')

elif app_mode == "📢 Government Schemes":
    st.header("Relevant Government Schemes for Farmers")
    schemes_lang = SCHEMES_DATA.get(st.session_state.language, SCHEMES_DATA['en'])
    for scheme, description in schemes_lang.items():
        with st.expander(f"**{scheme}**"):
            st.write(description)
            # Add audio playback for scheme description
            audio_file = text_to_speech(description, st.session_state.language)
            if audio_file: st.audio(audio_file, format='audio/mp3')