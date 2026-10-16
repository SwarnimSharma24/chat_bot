
from faiss_text import semantic_search, model, index, documents
from llm_service import generate_answer
from rag_context import build_context, build_prompt

query = "What is Python?"

# 1. Inspect retrieval with the same threshold as ask_chatbot()
results = semantic_search(
    query,
    model,
    index,
    documents,
    top_k=3,
    min_score=0.30
)

print("\n--- RETRIEVED RESULTS ---")
print(results)

# 2. Build the context
context = build_context(results)

print("\n--- CONTEXT ---")
print(context)

# 3. Build the exact prompt
prompt = build_prompt(query, context)

print("\n--- PROMPT ---")
print(prompt)

# 4. Call Ollama directly with that prompt
print("\n--- OLLAMA ANSWER ---")
print(generate_answer(prompt))
