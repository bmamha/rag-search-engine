import argparse
from lib.prompts import (
    augmented_generation_prompt,
    citations_prompt,
    summarize_prompt,
)
from lib.hybrid_search import rrf_hybrid_search
from lib.llm_requests import llm_response_generator
from lib.augmented_utils import augmented_text


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize our rrf search results using llm"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarizer")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Limit for the number of search results"
    )

    citation_parser = subparsers.add_parser(
        "citations",
        help="command to request LLM include citations during summary of search results",
    )
    citation_parser.add_argument("query", type=str, help="Search query")
    citation_parser.add_argument(
        "--limit", type=int, default=5, help="Limit for search results"
    )
    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            # do RAG stuff here
            docs = rrf_hybrid_search(query)
            response = llm_response_generator(query, docs, augmented_generation_prompt)
            augmented_text(docs, "RAG Response:", response)
        case "summarize":
            query = args.query
            limit = args.limit
            docs = rrf_hybrid_search(query, limit=limit)
            response = llm_response_generator(query, docs, summarize_prompt)
            augmented_text(docs, "LLM Summary:", response)
        case "citations":
            query = args.query
            limit = args.limit
            docs = rrf_hybrid_search(query, limit=limit)
            response = llm_response_generator(query, docs, citations_prompt)
            augmented_text(docs, "LLM Answer:", response)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
