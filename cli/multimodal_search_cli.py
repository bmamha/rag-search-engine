import argparse

from lib.multimodal_search import image_search_command, verify_image_embedding


parser = argparse.ArgumentParser("Multimodal Search CLI")

subparsers = parser.add_subparsers(dest="command", help="Available commands")

verify_parser = subparsers.add_parser(
    "verify_image_embedding", help="Verify image embedding is functional"
)

verify_parser.add_argument("file_path", type=str, help="file location of image")

image_parser = subparsers.add_parser(
    "image_search", help="Search documents using image"
)

image_parser.add_argument("file_path", type=str, help="location of image file")

args = parser.parse_args()


def main():
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.file_path)
        case "image_search":
            results = image_search_command(args.file_path)
            for i, doc in enumerate(results, 1):
                print(
                    f"{i}. {doc['title']} (similarity: {doc['similarity_score']:.3f})\n"
                )
                print(f"{doc['description'][:100]}...\n\n")


if __name__ == "__main__":
    main()
