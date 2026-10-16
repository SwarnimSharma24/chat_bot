
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Load the same embedding model used to build the index.
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)


# 2. Load the chunk text that matches the indexed vectors.
with open("chunks.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

if not isinstance(documents, list):
    raise ValueError("chunks.json must contain a list.")

if not all(isinstance(doc, str) and doc.strip() for doc in documents):
    raise ValueError("Every chunk must be a non-empty string.")


# 3. Load the persistent FAISS index.
index = faiss.read_index("documents.index")


# 4. Verify that each vector has a matching chunk.
if index.ntotal != len(documents):
    raise ValueError(
        f"Index contains {index.ntotal} vectors, "
        f"but chunks.json contains {len(documents)} chunks. "
        "Rebuild the index before searching."
    )

if index.d != 384:
    raise ValueError(
        f"Expected embedding dimension 384, got {index.d}."
    )


# 5. Search the index using semantic similarity.
def semantic_search(
    query,
    model,
    index,
    documents,
    top_k=3,
    min_score=0.4
):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if min_score is not None and not -1 <= min_score <= 1:
        raise ValueError("min_score must be between -1 and 1.")

    if index.ntotal == 0:
        return []

    # Convert the question into a normalized embedding vector.
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_embedding = np.ascontiguousarray(
        query_embedding,
        dtype=np.float32
    )

    # Retrieve the nearest matching chunk vectors.
    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, position in zip(scores[0], indices[0]):
        if position == -1:
            continue

        score = float(score)

        if min_score is not None and score < min_score:
            continue

        results.append({
            "document": documents[position],
            "score": score
        })

    return results


# 6. Test retrieval from the terminal.
if __name__ == "__main__":
    print(f"Chunks loaded: {len(documents)}")
    print(f"Vectors loaded: {index.ntotal}")
    print(f"Embedding dimension: {index.d}")

    query = input("\nAsk a question: ").strip()

    for top_k in (1, 3, 5):
        results = semantic_search(
            query,
            model,
            index,
            documents,
            top_k=top_k,
            min_score=0.30
        )

        print(f"\n--- Top {top_k} Results ---")

        if not results:
            print("No chunks met the minimum similarity score.")
            continue

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. {result['document']} "
                f"| Score: {result['score']:.4f}"
            )
