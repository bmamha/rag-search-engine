import os

from .chunked_semantic_search import ChunkedSemanticSearch
from .hybrid_utils import normalize, hybrid_score, rrf_score
from .inverted_index import InvertedIndex
from .search_utils import load_movies


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.doc_lengths_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=500):
        bm25_results = self._bm25_search(query, 500)
        semantic_search_results = self.semantic_search.search_chunks(query, 500)
        # we create a list of scores from both bm25 and semantic search results
        bm25_scores = bm25_results.values()
        semantic_scores = [x["score"] for x in semantic_search_results]
        # we normalize the scores to be between 0 and 1
        norm_bm25_scores = normalize(bm25_scores)
        norm_semantic_scores = normalize(semantic_scores)
        hybrid_score_dictionary = {}
        for i, doc_id in enumerate(bm25_results):
            hybrid_score_dictionary[doc_id] = self.idx.docmap[doc_id]
            hybrid_score_dictionary[doc_id]["keyword_score"] = norm_bm25_scores[i]
            hybrid_score_dictionary[doc_id]["semantic_score"] = 0.0

        for i, result in enumerate(semantic_search_results):
            doc_id = result["id"]
            if doc_id not in hybrid_score_dictionary:
                hybrid_score_dictionary[doc_id] = self.idx.docmap[doc_id]
                hybrid_score_dictionary[doc_id]["semantic_score"] = (
                    norm_semantic_scores[i]
                )
                hybrid_score_dictionary[doc_id]["keyword_score"] = 0.0
            else:
                hybrid_score_dictionary[doc_id]["semantic_score"] = (
                    norm_semantic_scores[i]
                )

        for key in hybrid_score_dictionary:
            hybrid_score_dictionary[key]["hybrid_score"] = hybrid_score(
                hybrid_score_dictionary[key]["keyword_score"],
                hybrid_score_dictionary[key]["semantic_score"],
                alpha,
            )

        sorted_results = [
            {k: v}
            for k, v in sorted(
                hybrid_score_dictionary.items(),
                key=lambda item: item[1]["hybrid_score"],
                reverse=True,
            )
        ]
        return sorted_results[:limit]

    def rrf_search(self, query, k, limit=10):
        bm25_results = self._bm25_search(query, 500)
        semantic_results = self.semantic_search.search_chunks(query, 500)
        rrf_dictionary = {}
        for i, result in enumerate(bm25_results.items()):
            rrf_dictionary[result[0]] = self.idx.docmap[result[0]]
            bm25_rank = i + 1
            bm25_rrf_score = rrf_score(bm25_rank, k)
            rrf_dictionary[result[0]]["bm25_rank"] = bm25_rank
            rrf_dictionary[result[0]]["bm25_rrf_score"] = bm25_rrf_score
            rrf_dictionary[result[0]]["semantic_rrf_score"] = 0.0
            rrf_dictionary[result[0]]["semantic_rank"] = -1

        for i, results in enumerate(semantic_results):
            semantic_rank = i + 1
            doc_id = results["id"]
            if rrf_dictionary.get(doc_id) is None:
                rrf_dictionary[doc_id] = self.idx.docmap[doc_id]
                rrf_dictionary[doc_id]["bm25_rrf_score"] = 0.0
                rrf_dictionary[doc_id]["bm25_rank"] = -1

            semantic_rrf_score = rrf_score(semantic_rank, k)
            rrf_dictionary[doc_id]["semantic_rank"] = semantic_rank
            rrf_dictionary[doc_id]["semantic_rrf_score"] = semantic_rrf_score

        for id in rrf_dictionary.keys():
            rrf_dictionary[id]["rrf_score"] = (
                rrf_dictionary[id]["bm25_rrf_score"]
                + rrf_dictionary[id]["semantic_rrf_score"]
            )

        sorted_rrf_results = dict(
            sorted(
                rrf_dictionary.items(), reverse=True, key=lambda x: x[1]["rrf_score"]
            )[:limit]
        )

        return sorted_rrf_results


def weighted_hybrid_search(query, alpha=0.5, limit=5):
    documents = load_movies()
    instance = HybridSearch(documents)
    return instance.weighted_search(query, alpha, limit)


def rrf_hybrid_search(query, k=60, limit=5):
    documents = load_movies()
    instance = HybridSearch(documents)
    return instance.rrf_search(query, k, limit)
