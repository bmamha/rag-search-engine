import os
import numpy as np
from .search_utils import load_movies
from .semantic_utils import cosine_similarity
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.documents_map = {}

    def generate_embedding(self, text):
        if text == "" or text.isspace():
            raise ValueError("Input text must be a non-empty string without spaces.")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embedding(self, documents):
        self.documents = documents
        for doc in documents:
            self.documents_map[doc["id"]] = doc

        doc_text = []
        for doc in documents:
            doc_text.append(f"{doc["title"]}: {doc["description"]}")

        self.embeddings = self.model.encode(doc_text, show_progress_bar=True)
        os.makedirs(os.path.dirname("cache/embeddings.npy"), exist_ok=True)
        np.save("cache/embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.documents_map[doc["id"]] = doc
        if os.path.exists("cache/embeddings.npy"):
            self.embeddings = np.load("cache/embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embedding(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        similarity_score_list = []
        for i in range(len(self.documents)):
            similarity_score = cosine_similarity(query_embedding, self.embeddings[i])
            similarity_score_list.append((similarity_score, self.documents[i]))

        similarity_score_list.sort(key=lambda x: x[0], reverse=True)

        query_results = []
        i = 0

        while limit > i:
            result = {}
            result["movie_idx"] = similarity_score_list[i][1]["id"]
            result["score"] = similarity_score_list[i][0]
            result["title"] = similarity_score_list[i][1]["title"]
            result["description"] = similarity_score_list[i][1]["description"]
            query_results.append(result)
            i += 1

        return query_results


def verify_embeddings():
    instance = SemanticSearch()
    documents = load_movies()
    embeddings = instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First three dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
