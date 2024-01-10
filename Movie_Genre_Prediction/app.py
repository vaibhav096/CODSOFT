import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# encoder = LabelEncoder()

# Load your TF-IDF vectorizer
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
import os
import pickle

# Get the absolute path of the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the pickle file
pickle_path = os.path.join(script_dir, 'vectorizer.pkl')

# Load the pickled object
tfidf = pickle.load(open(pickle_path, 'rb'))
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

