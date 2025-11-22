import json
import os
import string
from nltk.stem import PorterStemmer


DEFAULT_SEARCH_LIMIT = 5
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        movie_dictionary = json.load(f)
    return movie_dictionary["movies"]


def load_stop_words() -> list[str]:
    with open(STOP_WORDS_PATH, "r") as f:
        stop_words = f.read().splitlines()
    return stop_words


def preprocess_text(text: str) -> str:
    mytable = str.maketrans("", "", string.punctuation)
    return text.lower().translate(mytable)


def tokenize(text: str) -> list[str]:
    tokens = text.split()
    stop_words = load_stop_words()
    cleaned_tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in cleaned_tokens]
    return stemmed_tokens
