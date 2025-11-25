import math
from .search_utils import load_movies, PROJECT_ROOT, tokenize, preprocess_text
import os
import pickle
from collections import Counter

CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
CACHE_INDEX_PATH = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
CACHE_DOCMAP_PATH = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")
CACHE_TF_PATH = os.path.join(PROJECT_ROOT, "cache", "term_frequencies.pkl")


class InvertedIndex:
    index = {}
    docmap = {}
    term_frequencies = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(preprocess_text(text))
        cnt = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            cnt[token] += 1
        self.term_frequencies[doc_id] = cnt

    def get_documents(self, term: str) -> list[int]:
        return sorted(list(self.index.get(term.lower(), {})))

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)

    def save(self) -> None:
        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)

        with open(CACHE_INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)

        with open(CACHE_DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(CACHE_TF_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        try:
            with open(CACHE_INDEX_PATH, "rb") as f:
                self.index = pickle.load(f)
        except FileNotFoundError:
            raise Exception("File not found")

        try:
            with open(CACHE_DOCMAP_PATH, "rb") as f:
                self.docmap = pickle.load(f)
        except FileNotFoundError:
            raise Exception("File not found")

        try:
            with open(CACHE_TF_PATH, "rb") as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError:
            raise Exception("File not found")

    def get_term_frequencies(self, doc_id: int, term) -> int:
        token = tokenize(preprocess_text(term))
        if len(token) > 1:
            raise (Exception("Only single term is allowed"))

        return self.term_frequencies[doc_id][token[0]]

    def get_bm25_idf(self, term: str) -> float:
        token = tokenize(preprocess_text(term))
        if len(token) > 1:
            raise (Exception("Only single term is allowed"))
        N = len(self.docmap)
        df = len(self.get_documents(token[0]))
        bm25 = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25
