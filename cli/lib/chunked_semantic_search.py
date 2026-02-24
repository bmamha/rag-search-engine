import os
import json
import numpy as np
from lib.search_utils import semantic_chunk
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
            self.chunk_embdeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            return self.chunk_embdeddings

        return self.build_chunk_embeddings(documents)

