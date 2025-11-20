from .preprocess_text import preprocess_text
from .tokenize import tokenize
from .search_utils import load_movies, PROJECT_ROOT
import os
import pickle

CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
CACHE_INDEX_PATH = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
CACHE_DOCMAP_PATH = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")


class InvertedIndex:
    index = {}
    docmap = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(preprocess_text(text))
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

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
