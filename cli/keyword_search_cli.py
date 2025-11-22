#!/usr/bin/env python3
from lib.search import search
from lib.inverted_index import InvertedIndex
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser = subparsers.add_parser("build", help="Build inverted index")
    tf_parser = subparsers.add_parser("tf", help="Display term frequencies")

    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser.add_argument(
        "ID", type=int, help="Document ID to display term frequencies for"
    )
    tf_parser.add_argument("term", type=str, help="Term to display frequency for")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            results = search(args.query)
            for i, res in enumerate(results, start=1):
                print(f"{i}. `{res['title']} {res['id']}\n")
        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()

        case "tf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            frequency = inverted_index.get_term_frequencies(args.ID, args.term)
            print(
                f"Term Frequency of '{args.term}' in Document ID {args.ID}: {frequency}"
            )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
