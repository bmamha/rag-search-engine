import argparse

from lib.hybrid_utils import normalize_text
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies
from lib.enhance import enhance


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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_text(args.values)
        case "weighted-search":
            documents = load_movies()
            instance = HybridSearch(documents)
            weighted_results = instance.weighted_search(
                args.query, args.alpha, args.limit
            )
            for i, result in enumerate(weighted_results):
                for val in result.values():
                    print(
                        f"{i+1}. {val['title']}\nHybrid Score: {val['hybrid_score']:.4f}\nBM25 Score: {val['keyword_score']:.4f}\nSemantic Score: {val['semantic_score']:.4f}\n{val['description'][:100]}...\n"
                    )
        case "rrf-search":
            query = enhance(args.query, args.enhance) if args.enhance else args.query
            if query != args.query:
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
            documents = load_movies()
            instance = HybridSearch(documents)
            rrf_results = instance.rrf_search(query, args.k, args.limit)
            i = 0
            for result in rrf_results.values():
                print(
                    f"{i+1}. {result['title']}\nRRF Score: {result['rrf_score']:.4f}\nBM25Rank: {result['bm25_rank']}, Semantic Rank {result['semantic_rank']}\n{result['description'][:100]}...\n"
                )
                i += 1

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
