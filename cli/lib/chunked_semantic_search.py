import os
import json
import numpy as np
from lib.semantic_utils import cosine_similarity, semantic_chunk, SCORE_PRECISION
from .semantic_search import SemanticSearch


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.document_map = {}

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        doc_chunks = []
        chunk_metadata = []

        for i, doc in enumerate(documents):
            if doc["description"] == "":
                continue

            chunks = semantic_chunk(doc["description"], 4, 1)
            doc_chunks.extend(chunks)

            for j in range(len(chunks)):
                metadata = {}
                metadata["movie_idx"] = i
                metadata["chunk_id"] = j
                metadata["total_chunks"] = len(chunks)
                chunk_metadata.append(metadata)

        print(f"Chunks length: {len(doc_chunks)}")
        self.chunk_embdeddings = self.model.encode(doc_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname("cache/chunk_embeddings.npy"), exist_ok=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embdeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(doc_chunks)},
                f,
                indent=2,
            )
        return self.chunk_embdeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists(
            "cache/chunk_metadata.json"
        ):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(chunk_embedding, query_embedding)
            metadata = self.chunk_metadata[i]
            chunk_idx = metadata["chunk_id"]
            movie_idx = metadata["movie_idx"]

            chunk_scores.append(
                {
                    "chunk_idx": chunk_idx,
                    "movie_idx": movie_idx,
                    "score": score,
                }
            )

        movie_scores = {}

        for cs in chunk_scores:
            if (
                cs["movie_idx"] not in movie_scores
                or cs["score"] > movie_scores[cs["movie_idx"]]
            ):
                movie_scores[cs["movie_idx"]] = cs["score"]

        sorted_movie_scores = sorted(
            movie_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_movies_list = sorted_movie_scores[:limit]

        return_list = []

        for movies in top_movies_list:
            film_data = {}
            film_data["id"] = self.documents[movies[0]]["id"]
            film_data["title"] = self.documents[movies[0]]["title"]
            film_data["document"] = self.documents[movies[0]]["description"][:100]
            film_data["score"] = round(movies[1], SCORE_PRECISION)
            film_data["metadata"] = self.documents[movies[0]].get("metadata", {})
            return_list.append(film_data)

        return return_list
