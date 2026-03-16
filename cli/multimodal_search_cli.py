import argparse
from lib.multimodal_search import verify_image_embedding


parser = argparse.ArgumentParser("Multimodal Search CLI")

subparsers = parser.add_subparsers(dest="command", help="Available commands")

verify_parser = subparsers.add_parser(
    "verify_image_embedding", help="Verify image embedding is functional"
)

verify_parser.add_argument("file_path", type=str, help="file location of image")

args = parser.parse_args()


def main():
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.file_path)


if __name__ == "__main__":
    main()
