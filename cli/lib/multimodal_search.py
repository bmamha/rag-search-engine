import os
import pickle
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from .search_utils import PROJECT_ROOT, load_movies
from .semantic_utils import cosine_similarity


class MultiModalSearch:
    def __init__(self, documents: list, model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in documents:
            text = f"{doc['title']}: {doc['description']}"
            self.texts.append(text)
        self.text_embeddings = None

    def embed_image(self, image_path: str):
        file_path = os.path.join(PROJECT_ROOT, image_path)
        embedding = self.model.encode(Image.open(file_path))
        return embedding

    def load_or_create_embeddings(self, documents):
        print("creating or loading embeddings")
        if os.path.exists("cache/text_embeddings.npy"):
            print("Loading text embeddings")
            self.text_embeddings = np.load("cache/text_embeddings.npy")
            if len(self.text_embeddings) == len(documents):
                return self.text_embeddings
        return self.build_embedding()

    def build_embedding(self):
        print("Building text embedding from documents")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        os.makedirs(os.path.dirname("cache/text_embeddings.npy"), exist_ok=True)
        np.save("cache/text_embeddings.npy", self.text_embeddings)
        return self.text_embeddings

    def search_with_image(self, image_path: str):
        documents = self.documents
        image_embedding = self.embed_image(image_path)
        print("Searching already started")
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity_score = cosine_similarity(text_embedding, image_embedding)
            documents[i]["similarity_score"] = similarity_score

        documents.sort(key=lambda doc: doc["similarity_score"], reverse=True)
        return documents[:5]


def verify_image_embedding(image_path: str):
    instance = MultiModalSearch([])
    embedding = instance.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str):
    documents = load_movies()
    instance = MultiModalSearch(documents)
    instance.load_or_create_embeddings(documents)
    print("Creating embeddings completed")
    results = instance.search_with_image(image_path)
    return results
