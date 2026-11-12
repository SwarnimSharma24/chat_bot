import json
import re
import math
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "chunks.json"
INDEX_PATH = BASE_DIR / "documents.index"

# 1. Load the embedding model.
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


# 2. Load chunks and their metadata.
if not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        f"chunks.json not found at {CHUNKS_PATH}. Please run 'python rebuild_index.py' to generate it."
    )

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    documents = json.load(file)


if not isinstance(documents, list):
    raise ValueError("chunks.json must contain a list.")

required_fields = (
    "chunk_id",
    "document_id",
    "chunk_index",
    "text",
    "source",
)

for position, doc in enumerate(documents):
    if not isinstance(doc, dict):
        raise ValueError(f"Chunk at position {position} must be a dictionary.")

    if not all(field in doc for field in required_fields):
        raise ValueError(f"Chunk at position {position} has missing metadata.")

    if not isinstance(doc["text"], str) or not doc["text"].strip():
        raise ValueError(f"Chunk at position {position} has invalid text.")

    for field in ("chunk_id", "document_id", "chunk_index"):
        if type(doc[field]) is not int or doc[field] < 0:
            raise ValueError(f"Chunk at position {position} has invalid {field}.")

    if not isinstance(doc["source"], str) or not doc["source"].strip():
        raise ValueError(f"Chunk at position {position} has invalid source.")

    if doc["chunk_id"] != position:
        raise ValueError(f"Chunk ID mismatch at position {position}.")


# 3. Load the persistent FAISS index.
if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"documents.index not found at {INDEX_PATH}. Please run 'python rebuild_index.py' to generate it."
    )
index = faiss.read_index(str(INDEX_PATH))


# 4. Verify the index and chunk mapping.
if index.ntotal != len(documents):
    raise ValueError(
        f"Index contains {index.ntotal} vectors, "
        f"but chunks.json contains {len(documents)} chunks. "
        "Rebuild the index before searching."
    )

if index.d != 384:
    raise ValueError(f"Expected embedding dimension 384, got {index.d}.")


# 5. Search using semantic similarity.
def semantic_search(
    query,
    model,
    index,
    documents,
    top_k=3,
    min_score=0.4,
):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if min_score is not None:
        if (
            isinstance(min_score, bool)
            or not isinstance(min_score, (int, float))
            or not math.isfinite(min_score)
        ):
            raise ValueError("min_score must be a finite number or None.")

        if not -1 <= min_score <= 1:
            raise ValueError("min_score must be between -1 and 1.")

    if index.ntotal == 0:
        return []

    # Encode the question using the same embedding model.
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_embedding = np.ascontiguousarray(
        query_embedding,
        dtype=np.float32,
    )

    # Search the nearest chunk vectors.
    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal),
    )

    results = []

    for score, position in zip(scores[0], indices[0]):
        if position == -1:
            continue

        score = float(score)

        if min_score is not None and score < min_score:
            continue

        chunk = documents[int(position)]

        results.append(
            {
                "document": chunk["text"],
                "score": score,
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"],
                "source": chunk["source"],
            }
        )

    return results


# 6. Normalize selected keyword tokens.
def normalize_token(word):
    if word.endswith("s") and word[:-1].endswith("ing"):
        return word[:-1]

    return word


# 7. Search using keyword matching.
def keyword_search(query, documents, top_k=3):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    # Common words that usually add little search value.
    stop_words = {
        "which",
        "what",
        "where",
        "when",
        "who",
        "is",
        "the",
        "a",
        "an",
        "does",
        "do",
        "how",
        "to",
        "of",
        "in",
        "for",
        "and",
        "or",
        "on",
        "with",
        "using",
        "me",
        "please",
        "our",
        "use",
        "it",
        "used",
        "why",
    }

    # Preserve technical identifiers such as llm_service.py.
    token_pattern = r"[\w]+(?:[.-][\w]+)*"

    query_words = {
        normalize_token(word)
        for word in re.findall(token_pattern, query.lower())
        if word not in stop_words
    }

    if not query_words:
        return []

    results = []

    for chunk in documents:
        chunk_text = chunk["text"].lower()

        # Match complete tokens, not substrings.
        chunk_words = {
            normalize_token(word) for word in re.findall(token_pattern, chunk_text)
        }

        matched_words = sorted(query_words & chunk_words)

        if not matched_words:
            continue

        results.append(
            {
                "document": chunk["text"],
                "score": len(matched_words) / len(query_words),
                "matched_words": matched_words,
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"],
                "source": chunk["source"],
            }
        )

    # Sort after checking every chunk, not inside the loop.
    results.sort(
        key=lambda item: (
            item["score"],
            len(item["matched_words"]),
        ),
        reverse=True,
    )

    return results[:top_k]


# 8. Combine semantic and keyword search using Weighted RRF.
def hybrid_search(
    query,
    model,
    index,
    documents,
    top_k=5,
    candidate_k=10,
    rrf_k=60,
    min_semantic_score=0.0,
    semantic_weight=1.0,
    keyword_weight=1.0,
):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if type(candidate_k) is not int or candidate_k < 1:
        raise ValueError("candidate_k must be a positive integer.")

    if type(rrf_k) is not int or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer.")

    # Validate semantic weight.
    if (
        isinstance(semantic_weight, bool)
        or not isinstance(semantic_weight, (int, float))
        or not math.isfinite(semantic_weight)
        or semantic_weight < 0
    ):
        raise ValueError("semantic_weight must be a finite non-negative number.")

    # Validate keyword weight.
    if (
        isinstance(keyword_weight, bool)
        or not isinstance(keyword_weight, (int, float))
        or not math.isfinite(keyword_weight)
        or keyword_weight < 0
    ):
        raise ValueError("keyword_weight must be a finite non-negative number.")

    if semantic_weight == 0 and keyword_weight == 0:
        raise ValueError("At least one retrieval weight must be greater than zero.")

    # Retrieve candidates from both search methods.
    semantic_results = semantic_search(
        query,
        model,
        index,
        documents,
        top_k=candidate_k,
        min_score=min_semantic_score,
    )

    keyword_results = keyword_search(
        query,
        documents,
        top_k=candidate_k,
    )

    # Store each chunk once and accumulate its RRF score.
    combined = {}

    # Add semantic retrieval contributions.
    for rank, result in enumerate(semantic_results, start=1):

        chunk_id = result["chunk_id"]

        combined[chunk_id] = {
            **result,
            "rrf_score": semantic_weight / (rrf_k + rank),
            "semantic_rank": rank,
            "keyword_rank": None,
            "semantic_score": result["score"],
            "keyword_score": None,
        }

    # Add keyword retrieval contributions.
    for rank, result in enumerate(keyword_results, start=1):

        chunk_id = result["chunk_id"]

        if chunk_id not in combined:

            combined[chunk_id] = {
                **result,
                "rrf_score": 0.0,
                "semantic_rank": None,
                "keyword_rank": rank,
                "semantic_score": None,
                "keyword_score": result["score"],
            }

        else:

            combined[chunk_id]["keyword_rank"] = rank
            combined[chunk_id]["keyword_score"] = result["score"]

        combined[chunk_id]["rrf_score"] += keyword_weight / (rrf_k + rank)

    # Highest combined RRF score first.
    results = sorted(
        combined.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return results[:top_k]


# 9. Test retrieval from the terminal.
if __name__ == "__main__":

    print(f"Chunks loaded: {len(documents)}")
    print(f"Vectors loaded: {index.ntotal}")
    print(f"Embedding dimension: {index.d}")

    query = input("\nAsk a question: ").strip()

    # Test semantic retrieval.
    for top_k in (1, 3, 5):

        results = semantic_search(
            query,
            model,
            index,
            documents,
            top_k=top_k,
            min_score=0.30,
        )

        print(f"\n--- Semantic Search: Top {top_k} ---")

        if not results:
            print("No chunks met the minimum similarity score.")
            continue

        for rank, result in enumerate(results, start=1):

            print(
                f"\n{rank}. {result['document']}\n"
                f"   Score: {result['score']:.4f}\n"
                f"   Source: {result['source']}\n"
                f"   Document ID: {result['document_id']}\n"
                f"   Chunk ID: {result['chunk_id']}\n"
                f"   Chunk index: {result['chunk_index']}"
            )

    # Test keyword retrieval.
    keyword_results = keyword_search(
        query,
        documents,
        top_k=5,
    )

    print("\n--- Keyword Search: Top 5 ---")

    if not keyword_results:

        print("No keyword matches found.")

    else:

        for rank, result in enumerate(keyword_results, start=1):

            print(
                f"\n{rank}. {result['document']}\n"
                f"   Score: {result['score']:.4f}\n"
                f"   Matched words: {result['matched_words']}\n"
                f"   Source: {result['source']}\n"
                f"   Chunk ID: {result['chunk_id']}"
            )
