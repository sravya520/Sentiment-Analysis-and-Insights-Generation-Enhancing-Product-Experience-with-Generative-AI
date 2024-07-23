import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import re
import string
import nltk
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the model and vectorizer
with open("svm_model.pkl", "rb") as file:
    model = pickle.load(file)
with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Download necessary NLTK data
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

# Text preprocessing functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

def remove_punctuation(text):
    return "".join([i for i in text if i not in string.punctuation])

def tokenization(text):
    return re.split(' ', text)

def remove_stopwords(text):
    return " ".join([i for i in text if i not in stopwords])

def lemmatizer(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.text not in set(stopwords)])

# Streamlit app
st.title("Comprehensive Guide on NLP")
st.markdown("By Dangeti Sravya")

# Display the main image
# image = Image.open("image.png")
# st.image(image, use_column_width=True)

# Text input from user
st.subheader("Enter your text here:")
user_input = st.text_area("")

# Generate word cloud
if user_input:
    user_input_cleaned = clean_text(user_input)
    wordcloud = WordCloud(stopwords=stopwords, background_color='white').generate(user_input_cleaned)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    user_input = remove_punctuation(user_input_cleaned)
    user_input = tokenization(user_input)
    user_input = remove_stopwords(user_input)
    user_input = lemmatizer(user_input)

# Predict sentiment
if st.button("Predict"):
    if user_input:
        text_vectorized = vectorizer.transform([" ".join(user_input)])
        prediction = model.predict(text_vectorized)[0]
        prediction_prob = model.decision_function(text_vectorized)[0]
        st.header("Prediction:")
        if prediction == -1:
            st.subheader("The sentiment of the given text is: Negative")
        elif prediction == 0:
            st.subheader("The sentiment of the given text is: Neutral")
        elif prediction == 1:
            st.subheader("The sentiment of the given text is: Positive")
        st.write(f"Confidence scores: {prediction_prob}")
    else:
        st.subheader("Please enter a text for prediction.")

# Display sentiment analysis image
# image = Image.open("sentimental analysis image.png")
# st.image(image, use_column_width=True)

# Feedback collection
st.subheader("Feedback")
feedback = st.radio("Was the prediction accurate?", ('Yes', 'No'))
if feedback:
    st.write("Thank you for your feedback!")
