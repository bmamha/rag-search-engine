import time
from .llm_requests import individual_rerank_score, batch_rerank, evaluate
from sentence_transformers import CrossEncoder


def normalize(values: list[float]) -> list[float]:
    maximum = max(values)
    minimum = min(values)

    if maximum == minimum:
        return [1.0 for _ in values]
    return [(x - minimum) / (maximum - minimum) for x in values]


def normalize_text(values: list[float]):
    normalized_values = normalize(values)
    if min(normalized_values) == 1.0:
        for x in normalized_values:
            print(f"- {x:.1f}\n")
        return
    for x in normalized_values:
        print(f"- {x:.4f}\n")


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank, k=60):
    return 1 / (k + rank)


def hybrid_result_text(results: dict):
    for i, result in enumerate(results.values()):
        print(
            f"{i+1}. {result['title']}\nRRF Score: {result['rrf_score']:.4f}\nBM25Rank: {result['bm25_rank']}, Semantic Rank {result['semantic_rank']}\n{result['description'][:100]}...\n"
        )


def rerank_results(query: str, method: str, results: dict):
    match method:
        case "individual":
            for id, doc in results.items():
                score = individual_rerank_score(query, doc)
                results[id]["individual_rerank_score"] = score
                time.sleep(5)
        case "batch":
            ranked_list = batch_rerank(query, results)
            for rank, id in enumerate(ranked_list, 1):
                results[id]["batch_rerank_score"] = rank
        case "cross_encoder":
            pairs = []
            for doc in results.values():
                pairs.append(
                    [
                        query,
                        f"{doc.get('title', '')} - {doc.get('description', '')}",
                    ]
                )
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            scores = cross_encoder.predict(pairs)
            counter = 0
            for id in results.keys():
                results[id]["cross_encoder_score"] = scores[counter]
                counter += 1

        case _:
            return


def reranked_results_text(limit: int, method: str, results: list, k: int):
    print(f"Re-ranking top {limit} results using {method} method...")
    print(
        f"Reciprocal Rank Fusion Results for 'family movie about bears in the woods' (k={k})"
    )

    for i, result in enumerate(results[:limit]):
        print(f"{i+1}. {result['title']}")
        if "cross_encoder_score" in result:
            print(f"Cross Encoder Score:{result['cross_encoder_score']:.4f}")
        print(f"RRF Score: {result['rrf_score']:.4f}")
        print(
            f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}"
        )
        print(f"{result['description'][:100]}...\n")


def evaluate_results(query: str, results: dict):
    score_list = evaluate(query, results)
    index = 0
    for id in results.keys():
        results[id]["relevant_rating"] = score_list[index]
        index += 1


def evaluate_results_text(query: str, results: dict):
    evaluate_results(query, results)
    i = 1
    for result in results.values():
        print(f"{i}. {result["title"]} {result["relevant_rating"]}/3\n")
        i += 1


def sort_rrf_results(method: str, doc: dict):
    match method:
        case "individual":
            sorted_results = sorted(
                doc.values(),
                key=lambda x: x.get("individual_rerank_score", 0),
                reverse=True,
            )
            return sorted_results
        case "batch":
            sorted_results = sorted(
                doc.values(), key=lambda x: x.get("batch_rerank_score", float("inf"))
            )
            return sorted_results
        case "cross_encoder":
            sorted_results = sorted(
                doc.values(),
                key=lambda x: x.get("cross_encoder_score", 0),
                reverse=True,
            )
            return sorted_results
        case _:
            return list(doc.values())
