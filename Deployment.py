import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import re
import string
import nltk
import spacy
from collections import Counter

# Load pre-trained model and vectorizer
with open("svm_model.pkl", "rb") as file:
    model = pickle.load(file)
with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

def clean_text(text):
    text = text.lower()
    return text.strip()

def remove_punctuation(text):
    return "".join([char for char in text if char not in string.punctuation])

def tokenization(text):
    return re.split(r'\s+', text)

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords]

def lemmatizer(tokens):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc if token.lemma_ not in stopwords]

def word_frequency(tokens):
    return Counter(tokens)

# Streamlit application layout
st.title("Comprehensive Guide on NLP")
st.markdown("By Dangeti Sravya")
image = Image.open("emoji_satisfaction_meter.jpg")
st.image(image, use_column_width=True)

st.subheader("Enter your text here:")
user_input = st.text_area("")

if user_input:
    cleaned_text = clean_text(user_input)
    cleaned_text = remove_punctuation(cleaned_text)
    tokens = tokenization(cleaned_text)
    tokens = remove_stopwords(tokens)
    lemmatized_tokens = lemmatizer(tokens)
    
    # Display word frequency
    st.subheader("Word Frequency:")
    word_freq = word_frequency(lemmatized_tokens)
    st.write(word_freq)

    if st.button("Predict"):
        text_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(text_vectorized)[0]
        st.header("Prediction:")
        if prediction == -1:
            st.subheader("The sentiment of the given text is: Negative")
        elif prediction == 0:
            st.subheader("The sentiment of the given text is: Neutral")
        elif prediction == 1:
            st.subheader("The sentiment of the given text is: Positive")
else:
    st.subheader("Please enter a text for prediction.")
