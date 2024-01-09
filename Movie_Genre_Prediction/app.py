import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# encoder = LabelEncoder()

# Load your TF-IDF vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
encoder = pickle.load(open('label_encoder.pkl', 'rb'))
# Load your model
model = pickle.load(open('model.pkl', 'rb'))

st.title('Genre Prediction')
description = st.text_input("Enter the description")

if st.button('Submit'):

    # Preprocess the input text
    transformed_description = tfidf.transform([description])

    # Make prediction
    prediction = model.predict(transformed_description)

    # Inverse transform to get the predicted genre
    result = encoder.inverse_transform(prediction)[0]

    # Display result
    st.header(f"Predicted Genre: {result}")

