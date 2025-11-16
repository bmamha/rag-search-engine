from .search_utils import load_movies, load_stop_words, DEFAULT_SEARCH_LIMIT
import string
from nltk.stem import PorterStemmer


def search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    preprocessed_query_token = tokenize(preprocess_text(query))
    for movie in movies:
        preprocessed_title_token = tokenize(preprocess_text(movie["title"]))
        if query_in_title(preprocessed_query_token, preprocessed_title_token):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def query_in_title(query: list[str], title: list[str]) -> bool:
    for query_token in query:
        for title_token in title:
            if query_token in title_token:
                return True

    return False


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
