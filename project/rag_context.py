
import re

from faiss_text import hybrid_search, model, index, documents
from llm_service import generate_answer
from test import extract_sources

FALLBACK = "I don't have enough information to answer that."

QUESTION_WORDS = {
    "what", "which", "who", "where", "when", "why", "how",
    "is", "are", "does", "do", "did", "can", "could",
    "would", "should", "will", "tell", "explain",
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
            if word.lower() not in [
                entity.lower() for entity in entities
            ]:
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
    Execute the RAG pipeline using configurable weighted hybrid retrieval.

    Weights can be changed for every individual query.
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

    # Step 1: Retrieve using weighted hybrid search.
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
    )

    print("\n--- DEBUG: HYBRID SEARCH RESULTS ---")
    print(search_results)

    # Step 2: Stop if no documents were retrieved.
    if not search_results:
        print("DEBUG: No relevant documents found.")
        return {"answer": FALLBACK, "sources": []}

    # Step 3: Build context.
    context = build_context(search_results)

    print("\n--- DEBUG: CONTEXT ---")
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

    # Step 5: Build the LLM prompt.
    prompt = build_prompt(query, context)

    print("\n--- DEBUG: FINAL PROMPT ---")
    print(prompt)

    # Step 6: Generate answer.
    answer = generate_answer(prompt)

    # Step 7: Extract source metadata.
    sources = extract_sources(search_results)

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
