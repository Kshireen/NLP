from flask import Flask, request, jsonify
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import asyncio
import pickle
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# Load the pre-trained TF-IDF vectorizer
with open('./SVM_model.pkl', 'rb') as file:
    svm = pickle.load(file)

# Load the pre-trained Bernoulli Naive Bayes model
with open('./tf_idf_word_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

# Preprocess the data
def preprocess_review(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    tokens = [
        lemmatizer.lemmatize(stemmer.stem(word), get_wordnet_pos(word))
        for word in tokens
        if word not in stop_words and word not in string.punctuation and not word.isdigit()
    ]

    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN,
                "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

app = Flask(__name__)

# POST route to get sentiment analysis using an asynchronus way
@app.route("/sentiment", methods=["POST"])
def Get_sentiment():
    try:
        data = request.get_json()
        preprocessed_review = preprocess_review(data['review'])
        tfidf_review = tfidf.transform([preprocessed_review])
        print(svm.predict(tfidf_review))
        sentiment_score = svm.predict(tfidf_review)[0]

        return jsonify({"score": str(sentiment_score), "review": data["review"]}), 201

    except KeyError:
        return jsonify({'message': 'Invalid data provided'}), 400
    except Exception as e:
        return jsonify({'message': str(e)}), 500

if __name__ == '__main__':
    app.run()
