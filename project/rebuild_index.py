
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import chunk_text


# 1. Load the original knowledge-base documents.
with open("documents.json", "r", encoding="utf-8") as file:
    documents = json.load(file)


# 2. Validate the original documents.
if not isinstance(documents, list) or not documents:
    raise ValueError("documents.json must contain a non-empty list.")

if not all(isinstance(doc, str) and doc.strip() for doc in documents):
    raise ValueError("Every document must be a non-empty string.")


# 3. Split each document into smaller, overlapping chunks.
chunks = []

for document_number, document in enumerate(documents, start=1):
    document_chunks = chunk_text(
        document,
        chunk_size=300,
        overlap=50
    )

    if not document_chunks:
        raise ValueError(
            f"Document {document_number} produced no chunks."
        )

    chunks.extend(document_chunks)


# 4. Make sure we have chunks to index.
if not chunks:
    raise ValueError("No chunks were generated.")


# 5. Load the same embedding model used by semantic search.
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)


# 6. Generate one embedding for every chunk.
embeddings = model.encode(
    chunks,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)

# FAISS expects a compatible numeric array.
embeddings = np.ascontiguousarray(
    embeddings,
    dtype=np.float32
)


# 7. Create the FAISS index.
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)


# 8. Add all chunk embeddings to the index.
index.add(embeddings)


# 9. Save the index and its matching chunk texts.
faiss.write_index(index, "documents.index")

with open("chunks.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)


# 10. Confirm that everything matches.
with open("chunks.json", "r", encoding="utf-8") as file:
    saved_chunks = json.load(file)

saved_index = faiss.read_index("documents.index")

assert saved_index.ntotal == len(saved_chunks)
assert saved_index.ntotal == len(chunks)

print("\nChunk indexing completed successfully!")
print(f"Original documents: {len(documents)}")
print(f"Generated chunks: {len(chunks)}")
print(f"Vectors indexed: {saved_index.ntotal}")
print(f"Vector dimension: {dimension}")
print("Saved files: chunks.json, documents.index")
