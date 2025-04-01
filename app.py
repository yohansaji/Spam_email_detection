import streamlit as st
import pickle
import numpy as np

# Load pickled models and vectorizers
with open("logistic_regression.pkl", "rb") as log_reg_file:
    log_reg_model = pickle.load(log_reg_file)
with open("nb_model.pkl", "rb") as nb_file:
    nb_model = pickle.load(nb_file)
with open("tfidf_vectorizer_logreg.pkl", "rb") as logreg_vec_file:
    logreg_vectorizer = pickle.load(logreg_vec_file)
with open("tfidf_vectorizer_nb.pkl", "rb") as nb_vec_file:
    nb_vectorizer = pickle.load(nb_vec_file)

# Function to preprocess text
def preprocess_text(text):
    return text.lower()

# Function to make prediction
def predict_spam(model, vectorizer, message):
    cleaned_msg = preprocess_text(message)
    vectorized_msg = vectorizer.transform([cleaned_msg]).toarray()
    prediction = model.predict(vectorized_msg)[0]
    probability = model.predict_proba(vectorized_msg)[0][prediction]
    return ("Spam" if prediction == 1 else "Ham", probability)

# Streamlit UI
st.title("Spam Message Classifier")
st.write("Enter a message to check if it's spam or not.")

message = st.text_area("Enter your message:")
model_choice = st.radio("Choose a model:", ["Logistic Regression", "Naïve Bayes"])

if st.button("Check Message"):
    if model_choice == "Logistic Regression":
        label, prob = predict_spam(log_reg_model, logreg_vectorizer, message)
    else:
        label, prob = predict_spam(nb_model, nb_vectorizer, message)
    
    st.write(f"Prediction: {label} ({prob:.2f} confidence)")