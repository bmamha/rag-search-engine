import json
import os
import string
from nltk.stem import PorterStemmer


DEFAULT_SEARCH_LIMIT = 5
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
BM25_K1 = 1.5
BM25_B = 0.75


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


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    overlapped_chunk = []
    for word in words:
        if len(current_chunk) < chunk_size:
            current_chunk.append(word)
            if len(current_chunk) > chunk_size - overlap:
                overlapped_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = overlapped_chunk + [word]
            overlapped_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
