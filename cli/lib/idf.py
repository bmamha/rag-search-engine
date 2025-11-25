from .search_utils import preprocess_text, tokenize
from .inverted_index import InvertedIndex

import math


def term_frequency(id: int, term: str) -> int:
    inverted_index = InvertedIndex()
    inverted_index.load()
    frequency = inverted_index.get_term_frequencies(id, term)
    return frequency


def idf(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    doc_count = len(inverted_index.docmap)
    token = tokenize(preprocess_text(term))
    if len(token) > 1:
        raise Exception("Please provide a single term for IDF calculation.")

    term_count = len(inverted_index.get_documents(token[0]))
    term_idf = math.log((doc_count + 1) / (term_count + 1))
    return term_idf


def tfidf(id: int, term: str) -> float:
    tf_idf = 0
    tokens = tokenize(preprocess_text(term))
    for token in tokens:
        tf_idf += term_frequency(id, token) * idf(token)
    return tf_idf


def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    bm25_idf = inverted_index.get_bm25_idf(term)
    return bm25_idf
