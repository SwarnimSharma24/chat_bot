
import re

from faiss_text import semantic_search, model, index, documents
from llm_service import generate_answer


FALLBACK = "I don't have enough information to answer that."

QUESTION_WORDS = {
    "what", "which", "who", "where", "when", "why", "how",
    "is", "are", "does", "do", "did", "can", "could",
    "would", "should", "will", "tell", "explain"
}


def build_context(search_results):
    """Convert retrieved documents into readable context."""

    if not search_results:
        return "No relevant information was found."

    context_parts = [
        result["document"]
        for result in search_results
    ]

    return "\n".join(context_parts)


def build_prompt(query, context):
    """Build the prompt that will be sent to the LLM."""

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
    """
    Learning baseline:
    Detect capitalized names and acronyms in the question
    that do not appear in the retrieved context.
    """

    words = re.findall(r"\b[A-Za-z][A-Za-z0-9]*\b", query)

    entities = []

    for position, word in enumerate(words):
        if word.lower() in QUESTION_WORDS:
            continue

        is_capitalized_name = (
            position > 0 and word[0].isupper()
        )

        is_acronym = (
            len(word) >= 2 and word.isupper()
        )

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
            flags=re.IGNORECASE
        )
    ]

    return missing_entities


def ask_chatbot(query):
    print("\n--- DEBUG: USER QUERY ---")
    print(repr(query))

    # Step 1: Retrieve candidate documents
    search_results = semantic_search(
        query,
        model,
        index,
        documents,
        top_k=3,
        min_score=0.30,
    )

    print("\n--- DEBUG: SEARCH RESULTS ---")
    print(search_results)

    # Step 2: Stop if retrieval returns nothing
    if not search_results:
        print("DEBUG: No documents passed the retrieval filter.")
        return FALLBACK

    # Step 3: Build the retrieved context
    context = build_context(search_results)

    print("\n--- DEBUG: CONTEXT ---")
    print(context)

    # Step 4: Check for missing named entities
    missing_entities = find_missing_named_entities(
        query,
        context,
    )

    print("\n--- DEBUG: MISSING ENTITIES ---")
    print(missing_entities)

    if missing_entities:
        print("DEBUG: Required entity missing from context.")
        return FALLBACK

    # Step 5: Build the LLM prompt
    prompt = build_prompt(query, context)

    print("\n--- DEBUG: FINAL PROMPT ---")
    print(prompt)

    # Step 6: Generate the answer
    answer = generate_answer(prompt)

    print("\n--- DEBUG: RAW ANSWER ---")
    print(repr(answer))

    return answer


if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()

        if query.lower() == "exit":
            print("Chatbot stopped.")
            break

        if not query:
            continue

        answer = ask_chatbot(query)

        print("\n--- FINAL CHATBOT ANSWER ---")
        print(answer)
