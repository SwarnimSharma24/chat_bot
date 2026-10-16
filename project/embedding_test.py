from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")


documents = [
    "Python is a programming language.",
    "Python is commonly used for web development and automation.",
    "JavaScript is used to build interactive websites.",
    "React is a JavaScript library for building user interfaces.",
    "The weather is very nice today.",
]

# Convert documents into vectors
document_embeddings = model.encode(documents)


def semantic_search(query, documents, document_embeddings, top_k=3):
    # Convert query into vector
    query_embedding = model.encode([query])

    # Calculate similarity
    similarities = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    # Combine documents with their similarity scores
    results = []

    for document, similarity in zip(documents, similarities):
        results.append({
            "document": document,
            "similarity": float(similarity)
        })

    # Sort by similarity
    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results[:top_k]


query = "What can I use Python for?"
query2 = "what is React?"

results1 = semantic_search(query, documents,document_embeddings)
results2 = semantic_search(query2, documents,document_embeddings)

print("\nSemantic Search Results")
print("----------------------")

print("\nQuery 1: What can I use Python for?")
print("------------------------------------")

for result in results1:
    print(
        f"{result['similarity']:.4f} → {result['document']}"
    )


print("\nQuery 2: What is React?")
print("------------------------")

for result in results2:
    print(
        f"{result['similarity']:.4f} → {result['document']}"
    )