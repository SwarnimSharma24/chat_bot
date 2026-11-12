def extract_sources(search_results):
    required_result = []

    for docs in search_results:
        required_result.append(
            {
                "chunk_id": docs["chunk_id"],
                "source": docs["source"],
                "score": docs["score"],
            }
        )
    return required_result
