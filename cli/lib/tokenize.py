from nltk.stem import PorterStemmer
from .search_utils import load_stop_words


def tokenize(text: str) -> list[str]:
    tokens = text.split()
    stop_words = load_stop_words()
    cleaned_tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in cleaned_tokens]
    return stemmed_tokens
