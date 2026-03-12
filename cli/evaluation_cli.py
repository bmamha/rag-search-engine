import argparse
import json

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here

    with open("data/golden_dataset.json", "r") as file:
        data = json.load(file)
    documents = load_movies()
    instance = HybridSearch(documents)
    precision_results = []
    test_cases = data["test_cases"]
    for tc in test_cases:
        total_retrieved = []
        relevant_retrieved = []
        query = tc["query"]
        search_results = instance.rrf_search(query, 60, limit)
        relevant_length = len(tc["relevant_docs"])
        for result in search_results.values():
            title = result["title"]
            total_retrieved.append(title)
            if title in tc["relevant_docs"]:
                relevant_retrieved.append(title)
        precision_results.append(
            {
                "query": query,
                "relevant_retrieved": relevant_retrieved,
                "total_retrieved": total_retrieved,
                "total_relevant": relevant_length,
            }
        )

    print(f"k={limit}")
    for result in precision_results:
        precision = len(result["relevant_retrieved"]) / len(result["total_retrieved"])
        recall = len(result["relevant_retrieved"]) / result["total_relevant"]
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"- Query: {result["query"]}")
        print("\n\n\n ")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        retrieved_films = ", ".join(result["total_retrieved"])
        relevant_films = ", ".join(result["relevant_retrieved"])
        print(f"  - Retrieved: {retrieved_films}")
        print(f"  - Relevant: {relevant_films}\n")


if __name__ == "__main__":
    main()
