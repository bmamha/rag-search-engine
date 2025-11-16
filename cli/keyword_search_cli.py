#!/usr/bin/env python3
from lib.search import search
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            results = search(args.query)
            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['title']}\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
