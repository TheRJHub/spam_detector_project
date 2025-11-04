import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Email Spam Detector")

# Use a session state to manage text input
if "text" not in st.session_state:
    st.session_state.text = ""

def clear_input():
    st.session_state.text = ""

# Input box
email_text = st.text_area("Enter your email message:", value=st.session_state.text, key="text")

if st.button("Check Spam"):
    sample = cv.transform([email_text])
    prediction = model.predict(sample)[0]
    if prediction == 1:
        st.error("🚨 This message is Spam!")
    else:
        st.success("✅ This message is Not Spam!")

    # Clear after result
    clear_input()
