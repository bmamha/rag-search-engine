from .search_utils import preprocess_text, tokenize
from .inverted_index import InvertedIndex


def term_frequency_command(id: int, term: str) -> int:
    inverted_index = InvertedIndex()
    inverted_index.load()
    frequency = inverted_index.get_term_frequencies(id, term)
    return frequency


def idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    term_idf = inverted_index.idf(term)
    return term_idf


def tfidf_command(id: int, term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    tf_idf = 0
    tokens = tokenize(preprocess_text(term))
    for token in tokens:
        tf_idf += inverted_index.get_term_frequencies(id, token) * inverted_index.idf(
            token
        )
    return tf_idf


def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    bm25_idf = inverted_index.get_bm25_idf(term)
    return bm25_idf
