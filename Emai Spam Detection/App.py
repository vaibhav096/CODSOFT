import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

nltk.data.path.append("/opt/render/nltk_data")
nltk.download('punkt', download_dir="/opt/render/nltk_data")
nltk.download('stopwords', download_dir="/opt/render/nltk_data")




def operations_on_text(text):
    # converting text to the lowercase
    text = text.lower()
    # tokenize or cutting down the text into words using nltk
    text = nltk.word_tokenize(text)
    # for removing the special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email Spam Detector')
input_sns = st.text_input("Enter the mail")

if st.button('Submit'):

    # 1.preprocess

    transformed_mail = operations_on_text(input_sns)

    # 2.vectorization
    vector_input = tfidf.transform([transformed_mail])

    # step 3 is of prediction

    result  = model.predict(vector_input)[0]

    # 4. display result
    if(result==1):
        st.header("Email Spam Detected")
    else:
        st.header("Not a Spam Email")