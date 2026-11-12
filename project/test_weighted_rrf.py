
semantic_results = [
    {"chunk_id": 40, "score": 0.85},
    {"chunk_id": 23, "score": 0.72},
    {"chunk_id": 42, "score": 0.65},
]

keyword_results = [
    {"chunk_id": 23, "score": 0.90},
    {"chunk_id": 42, "score": 0.75},
    {"chunk_id": 40, "score": 0.50},
]


def weighted_rrf(
    semantic_results,
    keyword_results,
    rrf_k=60,
    semantic_weight=1.0,
    keyword_weight=1.0,
):
    combined = {}

    for rank, result in enumerate(semantic_results, start=1):
        chunk_id = result["chunk_id"]

        combined[chunk_id] = {
            "semantic_rank": rank,
            "keyword_rank": None,
            "rrf_score": semantic_weight / (rrf_k + rank),
        }

    for rank, result in enumerate(keyword_results, start=1):
        chunk_id = result["chunk_id"]

        if chunk_id not in combined:
            combined[chunk_id] = {
                "semantic_rank": None,
                "keyword_rank": rank,
                "rrf_score": 0.0,
            }

        else:
            combined[chunk_id]["keyword_rank"] = rank

        combined[chunk_id]["rrf_score"] += (
            keyword_weight / (rrf_k + rank)
        )

    return sorted(
        combined.items(),
        key=lambda item: item[1]["rrf_score"],
        reverse=True,
    )


# Experiment with different configurations.
configurations = [
    (1.0, 1.0),
    (2.0, 1.0),
    (1.0, 2.0),
    (3.0, 1.0),
]

for semantic_weight, keyword_weight in configurations:

    results = weighted_rrf(
        semantic_results,
        keyword_results,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
    )

    print(
        f"\nSemantic={semantic_weight}, "
        f"Keyword={keyword_weight}"
    )

    for chunk_id, data in results:
        print(chunk_id, round(data["rrf_score"], 6))
