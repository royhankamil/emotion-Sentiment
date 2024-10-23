import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class Preprocessing:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def case_folding(self, text):
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\n+', '', text)
        text = re.sub(r'\r+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans("", "", string.punctuation))

        emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U00010000-\U0010ffff"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r"", text)

        return text.lower()

    def tokenize(self, text):
        return word_tokenize(text)

    def stopword(self, text):
        stopw = stopwords.words("english")
        return [word for word in text if word not in stopw]

    def stemming(self, text):
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in text]

    def preprocess_input(self, text):
        text = self.case_folding(text)
        tokens = self.tokenize(text)
        tokens = self.stopword(tokens)
        tokens = self.stemming(tokens)
        return " ".join(tokens)  # Join back to string format

    def transform_to_tfidf(self, input_text):
        # Preprocess the input
        preprocessed_text = self.preprocess_input(input_text)
        # Transform using the saved TF-IDF vectorizer
        tfidf_vector = self.vectorizer.transform([preprocessed_text])  # Transform as list of texts
        return tfidf_vector
    