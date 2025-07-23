import streamlit as st
import joblib
from utils import preprocess_text

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.set_page_config(page_title="SMS Spam Classifier", layout="centered")
st.title("📩 SMS Spam Classifier")

st.markdown("Enter your message below to check if it's spam or not:")

user_input = st.text_area("✉️ Type your SMS message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = preprocess_text(user_input)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        prob = model.predict_proba(vect)[0][1]

        if pred == 1:
            st.error(f"🚫 This is **SPAM** (Confidence: {prob:.2%})")
        else:
            st.success(f"✅ This is **HAM** (Not Spam) (Confidence: {1 - prob:.2%})")
