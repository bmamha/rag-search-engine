#!/usr/bin/env python3

import argparse

from numpy import char
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    SemanticSearch,
)
from lib.search_utils import load_movies, chunk_text


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify semantic search model upload works")
    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Embed a text string and display the first three dimensions"
    )
    embed_text_parser.add_argument(
        "text",
        type=str,
        help="Text string to embed (must be a single word without spaces)",
    )

    subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings generation and caching"
    )
    embed_query = subparsers.add_parser(
        "embedquery",
        help="Embed a query text string and display the first three dimensions",
    )
    embed_query.add_argument("query", type=str, help="Query text string to embed")

    search_parser = subparsers.add_parser(
        "search", help="Search for similar documents based on a query"
    )
    search_parser.add_argument(
        "query", type=str, help="Query text string to search for"
    )

    search_parser.add_argument(
        "--limit", type=int, help="Number of search results to return"
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunk a text string and display the chunks"
    )
    chunk_parser.add_argument("text", type=str, help="Text string to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Maximum number of characters per chunk (default: 200)",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="shared number of words accross chunks",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            instance = SemanticSearch()
            documents = load_movies()
            instance.load_or_create_embeddings(documents)
            query_results = instance.search(args.query, limit=args.limit)

            for i in range(len(query_results)):
                print(
                    f"{i+1}. {query_results[i]['title']} (score: {query_results[i]['score']:.4f})\n"
                )
                print(f"   {query_results[i]['description']}\n")
        case "chunk":
            print(f"Chunking {len(list(args.text))} characters")
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            rank = 0
            for chunk in chunks:
                print(f"{rank+1}. {chunk}\n")
                rank += 1

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
