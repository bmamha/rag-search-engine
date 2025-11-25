#!/usr/bin/env python3
from lib.search import search
from lib.inverted_index import InvertedIndex
from lib.search_utils import tokenize, preprocess_text
from lib.idf import bm25_idf_command, term_frequency, idf, tfidf
import argparse
import math


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Display term frequencies")
    tf_parser.add_argument(
        "ID", type=int, help="Document ID to display term frequencies for"
    )
    tf_parser.add_argument("term", type=str, help="Term to display frequency for")

    idf_parser = subparsers.add_parser(
        "idf", help="Display inverse document frequencies"
    )

    idf_parser.add_argument("term", type=str, help="Term to display IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Display TF-IDF scores")
    tfidf_parser.add_argument(
        "ID", type=int, help="Document ID to display TF-IDF scores for"
    )
    tfidf_parser.add_argument("term", type=str, help="Term to display TF-IDF score for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            results = search(args.query)
            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['title']} {res['id']}\n")
        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()

        case "tf":
            frequency = term_frequency(args.ID, args.term)
            print(
                f"Term Frequency of '{args.term}' in Document ID {args.ID}: {frequency}"
            )
        case "idf":
            term_idf = idf(args.term)
            print(f"Inverse Document Frequency of '{args.term}': {term_idf:.2f}")

        case "tfidf":
            tf_idf = tfidf(args.ID, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.ID}': {tf_idf:.2f}"
            )
        case "bm25idf":
            bm25 = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
