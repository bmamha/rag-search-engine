#!/usr/bin/env python3
from lib.search import search
from lib.inverted_index import InvertedIndex
from lib.idf import (
    bm25_idf_command,
    term_frequency_command,
    idf_command,
    tfidf_command,
    bm25_tf_command,
)
from lib.search_utils import BM25_K1, BM25_B, DEFAULT_SEARCH_LIMIT
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")

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

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit", type=int, help="Optional argument for number of results displayed"
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
            frequency = term_frequency_command(args.ID, args.term)
            print(
                f"Term Frequency of '{args.term}' in Document ID {args.ID}: {frequency}"
            )
        case "idf":
            term_idf = idf_command(args.term)
            print(f"Inverse Document Frequency of '{args.term}': {term_idf:.2f}")

        case "tfidf":
            tf_idf = tfidf_command(args.ID, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.ID}': {tf_idf:.2f}"
            )
        case "bm25idf":
            bm25 = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "bm25search":
            inverted_index = InvertedIndex()
            inverted_index.load()
            results = inverted_index.bm25_search(args.query, DEFAULT_SEARCH_LIMIT)
            print(len(results))
            print(args.limit)
            list_count = 1

            for doc_id in results:
                print(
                    f"{list_count}. ({doc_id}) {inverted_index.docmap[doc_id]['title']} - Score:  {results[doc_id]:.2f}\n"
                )
                list_count += 1
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
