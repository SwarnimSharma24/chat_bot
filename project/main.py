stop_words = {
    "what",
    "is",
    "the",
    "a",
    "an",
    "of",
    "in",
    "to",
    "for",
    "and",
    "or",
    "on",
    "how",
    "why",
    "was",
    "are",
    "this",
    "that",
}

stop_special_Characters = {
    "?",
    "!",
    ".",
    ",",
    ":",
    ";",
}


def create_document(title, content, source):
    return {"title": title, "content": content, "source": source}


python_basics = create_document(
    "Python Basics", "Python is a programming language.", "python.txt"
)

python_history = create_document(
    "Python History",
    "Python was created by Guido van Rossum.",
    "Python_history.txt",
)

llm_introduction = create_document(
    "LLM Introduction", "LLM stands for Large Language Model.", "llm.txt"
)

rag_introduction = create_document(
    "RAG Introduction", "RAG stands for Retrieval-Augmented Generation.", "rag.txt"
)

documents = [python_basics, python_history, llm_introduction, rag_introduction]


# GET REQUIRED KEYWORDS
def remove_stop_words(keyword):
    filtered_words = []

    for key in keyword.split():

        for character in stop_special_Characters:
            key = key.replace(character, "")

        key = key.lower()

        if key and key not in stop_words:
            filtered_words.append(key)

    return filtered_words


# CALCULATE SCORE
def calculate_score(doc, keywords):
    score = 0

    for keyword in keywords:
        if keyword in doc["title"].lower() or keyword in doc["content"].lower():
            score += 1
    return score


# DATA SEARCH FUNCTION
def search_documents(documents, keyword, top_k):
    exact_keywords = remove_stop_words(keyword)
    min_relevance = max(1, len(exact_keywords) // 2)

    scored_documents = []

    for doc in documents:
        score = calculate_score(doc, exact_keywords)

        if score >= min_relevance:
            scored_documents.append((doc, score))

    scored_documents.sort(key=lambda item: item[1], reverse=True)

    return scored_documents[: top_k]


print("🤖 Ai chat bot started")
print("Type exit to exit the chatbot")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("🤖 Bot: Goodbye!")
        break

    results = search_documents(documents, user_input, 2)

    if results:
        print("🤖 Bot: I found some relative information.")

        for doc, score in results:
            print(f"{doc['title']} (Relevance: {score})")
            print(f"{doc['content']}")

    else:
        print("🤖 Bot i couldn't find anything related to that")
