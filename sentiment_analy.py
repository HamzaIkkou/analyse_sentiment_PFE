import streamlit as st
from googletrans import Translator
import joblib
import re

# Load your trained model and vectorizer
model = joblib.load("sentiment_analysis_model.joblib")       # Replace with your actual path
vectorizer = joblib.load("tfidf_vectorizer.pkl")       # Replace with your actual path

# Initialize translator
translator = Translator()

# Text preprocessing (optional, adjust to your model’s needs)
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # lowercase and remove punctuation
    return text

# Translate to English and detect language
def translate_to_english(text):
    result = translator.translate(text, dest='en')
    return result.text, result.src

# Streamlit UI
st.title("🌍 Analyseur de Sentiment")

user_input = st.text_area("Entrez votre commentaire (dans n'importe quelle langue):")

if user_input:
    with st.spinner("Analyse en cours..."):
        try:
            translated_text, detected_lang = translate_to_english(user_input)
            processed_text = preprocess(translated_text)
            vectorized_input = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_input)[0]
            sentiment = "😊 Positif" if prediction == 1 else "☹️ Négatif"

            st.subheader("🔍 Résultat de l'analyse")
            st.write(f"**Langue détectée:** {detected_lang}")
            st.write(f"**Traduction du commentaire:** {translated_text}")
            st.write(f"**Sentiment :** {sentiment}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
