import streamlit as st
import pickle
import re

# Load files
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# UI
st.title("🧠 Frances' Sentiment Analysis App")
st.write("Enter text and find out if it's Positive or Negative.")

text_input = st.text_input("Type your sentence:")

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess(text_input)
        vect = vectorizer.transform([cleaned])
        result = model.predict(vect)[0]

        if result == 1:
            st.success("Prediction: Positive 😊")
        else:
            st.error("Prediction: Negative 😠")
