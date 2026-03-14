import argparse
from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch
from lib.llm_requests import augmented_generation


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            documents = load_movies()
            instance = HybridSearch(documents)
            docs = instance.rrf_search(query, 60, 5)
            response = augmented_generation(query, docs)
            print("Search Results")
            for doc in docs.values():
                print(f"- {doc['title']}")
            print("RAG response")
            print(response)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
