import argparse

from lib.hybrid_utils import (
    hybrid_result_text,
    normalize_text,
    sort_rrf_results,
    rerank_results,
    reranked_results_text,
    evaluate_results_text,
)
from lib.hybrid_search import rrf_hybrid_search, weighted_hybrid_search
from lib.llm_requests import enhance


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a text string"
    )
    normalize_parser.add_argument(
        "values", nargs="+", type=float, help="Values for the command"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Perform a weighted search"
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weighting factor for BM25 vs semantic score (default: 0.5)",
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=500, help="Number of search results to return"
    )

    rrf_parser = subparsers.add_parser(
        "rrf-search", help="Perform a Reciprocal Rank Fusion search"
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results to return"
    )

    rrf_parser.add_argument(
        "--k",
        type=int,
        default=60,
        help="controls how much more weight we give to higher ranked results",
    )

    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--evaluate",
        "-e",
        action="store_true",
        help="Enable LLM to evaluate the relevancy of our ranked search",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="re-rank results using LLMs",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_text(args.values)
        case "weighted-search":
            weighted_results = weighted_hybrid_search(
                args.query, args.alpha, args.limit
            )
            for i, result in enumerate(weighted_results):
                for val in result.values():
                    print(
                        f"{i+1}. {val['title']}\nHybrid Score: {val['hybrid_score']:.4f}\nBM25 Score: {val['keyword_score']:.4f}\nSemantic Score: {val['semantic_score']:.4f}\n{val['description'][:100]}...\n"
                    )
        case "rrf-search":
            limit = args.limit
            k = args.k
            query = enhance(args.query, args.enhance) if args.enhance else args.query
            if query != args.query:
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

            if args.rerank_method:
                limit = limit * 5
            rrf_results = rrf_hybrid_search(query, k, limit)
            hybrid_result_text(rrf_results)
            """the following additional steps execute if we request an 
             LLM-based re-ranking method"""
            method = args.rerank_method
            if method:
                rerank_results(query, method, rrf_results)
                sorted_rrf_results = sort_rrf_results(method, rrf_results)
                reranked_results_text(args.limit, method, sorted_rrf_results, k)
            if args.evaluate:
                evaluate_results_text(query, rrf_results)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
