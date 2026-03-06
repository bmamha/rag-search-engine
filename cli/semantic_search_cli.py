#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    SemanticSearch,
)

from lib.chunked_semantic_search import ChunkedSemanticSearch
from lib.search_utils import load_movies
from lib.semantic_utils import chunk_text, semantic_chunk_text


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

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Perform a semantic chunk."
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text string to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="Maximum size of chunk"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Number of shared words across chunks"
    )

    subparsers.add_parser("embed_chunks", help="Embed text chunks of movies data")

    search_chunked = subparsers.add_parser(
        "search_chunked", help="Search for similar documents using chunked embeddings"
    )
    search_chunked.add_argument(
        "query", type=str, help="Query text string to search for"
    )
    search_chunked.add_argument(
        "--limit", type=int, default=5, help="Number of search results to return"
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
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            instance = ChunkedSemanticSearch()
            documents = load_movies()
            embeddings = instance.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            instance = ChunkedSemanticSearch()
            documents = load_movies()
            embeddings = instance.load_or_create_chunk_embeddings(documents)
            results = instance.search_chunks(args.query, args.limit)
            for i, film in enumerate(results):
                print(f"\n{i+1}. {film["title"]} (score: {film["score"]:.4f})")
                print(f"   {film["document"]}...\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
