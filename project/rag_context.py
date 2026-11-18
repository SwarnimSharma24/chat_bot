import re

from faiss_text import hybrid_search, model, index, documents
from llm_service import generate_answer
from test import extract_sources
from sentence_transformers import CrossEncoder

# Load Cross-Encoder once when the module is imported.
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu",
)

FALLBACK = "I don't have enough information to answer that."

QUESTION_WORDS = {
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "does",
    "do",
    "did",
    "can",
    "could",
    "would",
    "should",
    "will",
    "tell",
    "explain",
}


def build_context(search_results):
    """Convert retrieved documents into readable context."""

    if not search_results:
        return "No relevant information was found."

    context_parts = []

    for result in search_results:
        document = result.get("document", "")
        document = re.sub(r"\s+", " ", document).strip()

        if document:
            context_parts.append(document)

    return "\n\n".join(context_parts)

def rerank_documents(query, search_results, top_k=3):
    """
    Rerank retrieved candidates using a Cross-Encoder.
    """

    if not search_results:
        return []

    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    # Create query document pairs.
    pairs = [
        (query, result["document"])
        for result in search_results
    ]

    # calculate relevance scores.
    scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )

    # Attact cross-encoder scores.
    reranked_results = []

    for result, score in zip(search_results, scores):

        reranked_result = {
            **result,
            "rerank_score": float(score),
        }

        reranked_results.append(reranked_result)

    # sort by cross-encoder relevance score
    reranked_results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return reranked_results[:top_k]


def build_prompt(query, context):
    """Build the prompt sent to the LLM."""

    prompt = f"""You are a helpful AI assistant.

Answer the user's question using only the information
provided in the CONTEXT.

Instructions:
1. Use only facts supported by the context.
2. Do not add facts from your general knowledge.
3. If the context does not contain enough information,
   reply exactly: {FALLBACK}
4. Keep your answer simple and concise.

CONTEXT:
<context>
{context}
</context>

QUESTION:
{query}

ANSWER:"""

    return prompt.strip()


def find_missing_named_entities(query, context):
    """Detect potentially missing capitalized names and acronyms."""

    words = re.findall(r"\b[A-Za-z][A-Za-z0-9]*\b", query)

    entities = []

    for position, word in enumerate(words):
        if word.lower() in QUESTION_WORDS:
            continue

        is_capitalized_name = position > 0 and word[0].isupper()
        is_acronym = len(word) >= 2 and word.isupper()

        if is_capitalized_name or is_acronym:
            if word.lower() not in [entity.lower() for entity in entities]:
                entities.append(word)

    missing_entities = [
        entity
        for entity in entities
        if not re.search(
            rf"\b{re.escape(entity)}\b",
            context,
            flags=re.IGNORECASE,
        )
    ]

    return missing_entities



def ask_chatbot(
    query,
    semantic_weight=3.0,
    keyword_weight=1.0,
):
    """
    Execute RAG using hybrid retrieval and Cross-Encoder reranking.
    """

    query = query.strip()

    if not query:
        return {"answer": FALLBACK, "sources": []}

    print("\n--- DEBUG: USER QUERY ---")
    print(repr(query))

    print("\n--- DEBUG: RETRIEVAL CONFIGURATION ---")
    print({
        "semantic_weight": semantic_weight,
        "keyword_weight": keyword_weight,
    })

    # Step 1: Retrieve all hybrid candidates.
    search_results = hybrid_search(
        query,
        model,
        index,
        documents,
        top_k=3,
        candidate_k=10,
        rrf_k=60,
        min_semantic_score=0.30,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        return_all_candidates=True,
    )

    print("\n--- DEBUG: HYBRID CANDIDATES ---")
    print(f"Total candidates: {len(search_results)}")

    if not search_results:
        return {"answer": FALLBACK, "sources": []}

    # Step 2: Rerank all retrieved candidates.
    reranked_results = rerank_documents(
        query,
        search_results,
        top_k=3,
    )

    print("\n--- DEBUG: RERANKED RESULTS ---")

    for rank, result in enumerate(reranked_results, start=1):
        print({
            "rank": rank,
            "chunk_id": result["chunk_id"],
            "rrf_score": result["rrf_score"],
            "rerank_score": result["rerank_score"],
        })

    # Step 3: Build context using final top 3.
    context = build_context(reranked_results)

    print("\n--- DEBUG: FINAL CONTEXT ---")
    print(context)

    # Step 4: Check for missing named entities.
    missing_entities = find_missing_named_entities(
        query,
        context,
    )

    print("\n--- DEBUG: MISSING ENTITIES ---")
    print(missing_entities)

    if missing_entities:
        print("DEBUG: Required entity missing from context.")
        return {"answer": FALLBACK, "sources": []}

    # Step 5: Build the final LLM prompt.
    prompt = build_prompt(query, context)

    # Step 6: Generate answer.
    answer = generate_answer(prompt)

    # Step 7: Extract sources from reranked documents.
    sources = extract_sources(reranked_results)

    print("\n--- DEBUG: RAW ANSWER ---")
    print(repr(answer))

    return {
        "answer": answer,
        "sources": sources,
    }



if __name__ == "__main__":

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()

        if query.lower() == "exit":
            print("Chatbot stopped.")
            break

        if not query:
            continue

        answer = ask_chatbot(
            query,
            semantic_weight=3.0,
            keyword_weight=1.0,
        )

        print("\n--- FINAL CHATBOT ANSWER ---")
        print(answer)
