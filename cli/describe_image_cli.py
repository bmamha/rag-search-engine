import argparse
import mimetypes
import os
from lib.search_utils import PROJECT_ROOT
from lib.prompts import IMAGE_PROMPT
from lib.llm_requests import llm_image_response_generator


IMAGE_PATH = os.path.join(PROJECT_ROOT, "data", "paddington.jpeg")
parser = argparse.ArgumentParser("Image description using LLMs")
parser.add_argument("--image", type=str, help="Path to image file")
parser.add_argument("--query", type=str, help="user query for image files")

args = parser.parse_args()


def main():
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(IMAGE_PATH, "rb") as file:
        img = file.read()

    response, tokens = llm_image_response_generator(img, IMAGE_PROMPT, mime, args.query)
    print(f"Rewritten query: {response}")
    if tokens:
        print(f"Total tokens:    {tokens}")


if __name__ == "__main__":
    main()
