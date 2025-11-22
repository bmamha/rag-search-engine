from .search_utils import preprocess_text, tokenize, DEFAULT_SEARCH_LIMIT
from .inverted_index import InvertedIndex


def search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    inverted_index = InvertedIndex()
    inverted_index.load()
    preprocessed_query_tokens = tokenize(preprocess_text(query))
    index_results = set()
    doc_results = []
    for token in preprocessed_query_tokens:
        token_doc_ids = inverted_index.get_documents(token)
        index_results.update(token_doc_ids)

    for index in sorted(index_results):
        doc_results.append(inverted_index.docmap[index])
        if len(doc_results) >= limit:
            break

    return doc_results
