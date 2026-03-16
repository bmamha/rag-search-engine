import os
from PIL import Image
from sentence_transformers import SentenceTransformer
from .search_utils import PROJECT_ROOT


class MultiModalSearch:
    def __init__(self, model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str):
        embedding = self.model.encode(Image.open(image_path))
        return embedding


def verify_image_embedding(image_file: str):
    file_path = os.path.join(PROJECT_ROOT, image_file)
    instance = MultiModalSearch()
    embedding = instance.embed_image(file_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
