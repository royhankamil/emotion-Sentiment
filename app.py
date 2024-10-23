import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
import os
import nltk
from Algorithm.algorithm import Preprocessing

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models/sentiment_model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'models/vectorizer.pkl')

model = load_pickle(model_path)
vectorizer = load_pickle(vectorizer_path)
preprocessing = Preprocessing(vectorizer)

def predict_sentiment(input_text):
    tfidf_vector = preprocessing.transform_to_tfidf(input_text)
    prediction = model.predict(tfidf_vector)[0]
    return prediction

def display_wordcloud(column_data, title, colormap="Blues_r"):
    st.write(title)
    wordcloud = WordCloud(
        height=800,
        width=1200,
        collocations=False,
        colormap=colormap,
        random_state=123
    ).generate(' '.join(column_data.dropna().to_list()))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

st.title("Emotion Sentiment Prediction")

st.subheader("Comment")
user_input = st.text_area("Input your comment here:")

if st.button("Predict"):
    if user_input:
        sentiment_code = predict_sentiment(user_input)
        sentiment_result = sentiment_code
        st.write(f"Sentimen: {sentiment_result}")
    else:
        st.write("No Input Here")