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
