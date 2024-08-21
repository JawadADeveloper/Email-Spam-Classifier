import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import string

# Initialize the PorterStemmer
ps = PorterStemmer()

# Load the pre-trained TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    text = word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]
    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    # Apply stemming
    text = [ps.stem(i) for i in text]

    return ' '.join(text)

# Create the Streamlit app
st.title('Email Spam Classifier')

# Input field for the message
input_sms = st.text_input('Enter the Message.')

if st.button('Predict'):
    # Preprocess the input text
    transform_sms = transform_text(input_sms)
    # Vectorize the preprocessed text
    vector_input = tfidf.transform([transform_sms])
    # Predict using the trained model
    result = model.predict(vector_input)[0]
    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
