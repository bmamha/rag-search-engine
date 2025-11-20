from .search_utils import load_movies, DEFAULT_SEARCH_LIMIT
from .tokenize import tokenize
from .inverted_index import InvertedIndex
from .preprocess_text import preprocess_text


def search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    inverted_index = InvertedIndex()
    inverted_index.load()
    preprocessed_query_tokens = tokenize(preprocess_text(query))
    index_results = set()
    doc_results = []
    for token in preprocessed_query_tokens:
        token_doc_ids = inverted_index.get_documents(token)
        index_results.update(token_doc_ids)

    print(len(index_results))

    for index in sorted(index_results):
        doc_results.append(inverted_index.docmap[index])
        if len(doc_results) >= limit:
            break

    return doc_results


def search_old(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
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
