import string
import numpy as np
import re


SCORE_PRECISION = 5


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_chunk(
    text: str,
    max_chunk_size: int,
    overlap: int,
) -> list[str]:
    stripped_text = text.strip()
    len(stripped_text)
    if stripped_text == "":
        return []
    sentences = re.split(r"(?<=[.!?])\s+", stripped_text)
    if len(sentences) == 1 and not sentences[0].endswith(tuple(string.punctuation)):
        sentences = [text]
    chunks = []
    i = 0
    n_sentences = len(sentences)
    print(
        f"maxum chunk size: {max_chunk_size}, overlap: {overlap}, total sentences: {n_sentences}"
    )

    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and (len(chunk_sentences) <= overlap):
            break
        cleaned_sentences = []
        for sentence in chunk_sentences:
            print(sentence)
            cleaned_sentences.append(sentence.strip())
        if not cleaned_sentences:
            continue
        chunk = " ".join(cleaned_sentences)
        chunks.append(chunk)
        i += max_chunk_size - overlap
    return chunks


def fixed_size_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    overlapped_chunk = []
    for word in words:
        if len(current_chunk) < chunk_size:
            current_chunk.append(word)
            if len(current_chunk) > chunk_size - overlap:
                overlapped_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = overlapped_chunk + [word]
            overlapped_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_text(text: str, max_chunk_size: int, overlap: int):
    print(f"Chunking {len(list(text))} characters\n")
    chunk_list = fixed_size_chunk(text, max_chunk_size, overlap)
    for i, chunk in enumerate(chunk_list):
        print(f"{i+1}. {chunk}\n")


def semantic_chunk_text(text: str, max_chunk_size: int, overlap: int):
    print(f"Semantic chunking {len(list(text))} characters\n")
    chunk_list = semantic_chunk(text, max_chunk_size, overlap)
    for i, chunk in enumerate(chunk_list):
        print(f"{i+1}. {chunk}\n")
