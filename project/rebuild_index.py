
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import chunk_text

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_PATH = BASE_DIR / "documents.json"
CHUNKS_PATH = BASE_DIR / "chunks.json"
INDEX_PATH = BASE_DIR / "documents.index"

# 1. Load the original knowledge-base documents.
with open(DOCUMENTS_PATH, "r", encoding="utf-8") as file:
    documents = json.load(file)


# 2. Validate the original documents.
if not isinstance(documents, list) or not documents:
    raise ValueError("documents.json must contain a non-empty list.")

if not all(isinstance(doc, str) and doc.strip() for doc in documents):
    raise ValueError("Every document must be a non-empty string.")


# 3. Create chunks and preserve their metadata.
chunks = []

for document_id, document in enumerate(documents):
    document_chunks = chunk_text(
        document,
        chunk_size=300,
        overlap=50
    )

    if not document_chunks:
        raise ValueError(
            f"Document {document_id} produced no chunks."
        )

    for chunk_index, chunk in enumerate(document_chunks):
        chunks.append({
            "chunk_id": len(chunks),
            "document_id": document_id,
            "chunk_index": chunk_index,
            "text": chunk,
            "source": "documents.json"
        })


# 4. Validate that chunks exist.
if not chunks:
    raise ValueError("No chunks were generated.")


# 5. Extract text for embedding.
# Metadata is not embedded; only the actual text is.
chunk_texts = [chunk["text"] for chunk in chunks]


# 6. Load the embedding model.
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)


# 7. Generate normalized embeddings.
embeddings = model.encode(
    chunk_texts,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)

embeddings = np.ascontiguousarray(
    embeddings,
    dtype=np.float32
)


# 8. Create and populate the FAISS index.
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)


# 9. Save the index and matching metadata records.
faiss.write_index(index, str(INDEX_PATH))

with open(CHUNKS_PATH, "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)


# 10. Validate the saved files.
with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    saved_chunks = json.load(file)

saved_index = faiss.read_index(str(INDEX_PATH))

assert saved_index.ntotal == len(saved_chunks)
assert saved_index.ntotal == len(chunks)

assert all(
    record["chunk_id"] == position
    for position, record in enumerate(saved_chunks)
)

print("\nChunk indexing completed successfully!")
print(f"Original documents: {len(documents)}")
print(f"Generated chunks: {len(chunks)}")
print(f"Vectors indexed: {saved_index.ntotal}")
print(f"Vector dimension: {dimension}")
print("Saved files: chunks.json, documents.index")
