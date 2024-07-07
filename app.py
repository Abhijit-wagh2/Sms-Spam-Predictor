import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.exceptions import NotFittedError
from scipy.sparse import csr_matrix

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load the saved TF-IDF Vectorizer and Model
try:
    with open('vectorizer.pkl', 'rb') as file:
        tfidf = pickle.load(file)
except (FileNotFoundError, pickle.UnpicklingError):
    st.error("Error loading TF-IDF Vectorizer. Please check the file and try again.")
    tfidf = None

try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except (FileNotFoundError, pickle.UnpicklingError):
    st.error("Error loading Model. Please check the file and try again.")
    model = None

# Streamlit App
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if tfidf is not None and model is not None:
        try:
            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)[0]
            # 4. Display
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
        except NotFittedError:
            st.error("TF-IDF Vectorizer is not fitted. Please check the vectorizer and try again.")
    else:
        st.error("Model or Vectorizer not loaded properly.")
