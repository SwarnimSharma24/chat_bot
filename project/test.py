# def extract_sources(search_results):
#     required_result = []

#     for docs in search_results:
#         required_result.append(
#             {
#                 "chunk_id": docs["chunk_id"],
#                 "source": docs["source"],
#                 "score": docs["score"],
#                 "semantic_score": docs.get("semantic_score"),
#                 "keyword_score": docs.get("keyword_score"),
#                 "rrf_score": docs.get("rrf_score"),
#                 "rerank_score": docs.get("rerank_score"),
#             }
#         )
#     return required_result


import time
import statistics

from faiss_text import hybrid_search, model, index, documents


def extract_sources(search_results):
    required_result = []

    for docs in search_results:
        required_result.append(
            {
                "chunk_id": docs["chunk_id"],
                "source": docs["source"],
                "score": docs["score"],
                "semantic_score": docs.get("semantic_score"),
                "keyword_score": docs.get("keyword_score"),
                "rrf_score": docs.get("rrf_score"),
                "rerank_score": docs.get("rerank_score"),
            }
        )

    return required_result


# --------------------------------------------------
# 1. EVALUATION METRICS
# --------------------------------------------------


def reciprocal_rank(retrieved_ids, relevant_ids, k=3):
    """
    Calculate Reciprocal Rank at K.
    """

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):

        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0


def precision_at_k(retrieved_ids, relevant_ids, k=3):
    """
    Calculate Precision at K.
    """

    top_k_ids = retrieved_ids[:k]

    if not top_k_ids:
        return 0.0

    relevant_count = sum(1 for chunk_id in top_k_ids if chunk_id in relevant_ids)

    return relevant_count / k


def recall_at_k(retrieved_ids, relevant_ids, k=3):
    """
    Calculate Recall at K.
    """

    if not relevant_ids:
        return 0.0

    top_k_ids = retrieved_ids[:k]

    relevant_count = sum(1 for chunk_id in top_k_ids if chunk_id in relevant_ids)

    return relevant_count / len(relevant_ids)


def evaluate_ranking(retrieved_ids, relevant_ids, k=3):
    """
    Calculate all retrieval metrics.
    """

    return {
        "mrr": reciprocal_rank(retrieved_ids, relevant_ids, k),
        "precision": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall": recall_at_k(retrieved_ids, relevant_ids, k),
    }


# --------------------------------------------------
# 2. RETRIEVAL EVALUATION
# --------------------------------------------------


def run_retrieval_evaluation():

    # Local import avoids circular import with rag_context.py.
    from rag_context import rerank_documents

    evaluation_dataset = [
        # 1. RAG fundamentals
        {
            "question": "What is RAG?",
            "relevant_ids": [23, 24],
        },
        # 2. Generic RAG pipeline
        {
            "question": "How does a RAG pipeline work?",
            "relevant_ids": [25],
        },
        # 3. Our chatbot's complete pipeline
        {
            "question": "How does our chatbot process a question from document ingestion to answer generation?",
            "relevant_ids": [49, 50, 51],
        },
        # 4. FAISS
        {
            "question": "What is FAISS?",
            "relevant_ids": [30],
        },
        # 5. Semantic search
        {
            "question": "What is semantic search?",
            "relevant_ids": [28],
        },
        # 6. Embeddings
        {
            "question": "What are embeddings?",
            "relevant_ids": [27],
        },
        # 7. Sentence Transformers
        {
            "question": "What is Sentence Transformers used for?",
            "relevant_ids": [31, 32],
        },
        # 8. Chunking
        {
            "question": "What is the purpose of chunking?",
            "relevant_ids": [40],
        },
        # 9. Chatbot retrieval
        {
            "question": "How does our chatbot retrieve relevant information?",
            "relevant_ids": [31, 49, 50],
        },
        # 10. Ollama
        {
            "question": "What is Ollama?",
            "relevant_ids": [19],
        },
    ]

    TOP_K = 3
    REPETITIONS = 10

    CANDIDATE_K_VALUES = [10, 15, 20]

    experiment_results = {}

    # --------------------------------------------------
    # 1. INITIALIZE EXPERIMENTS
    # --------------------------------------------------

    for candidate_k in CANDIDATE_K_VALUES:

        experiment_results[candidate_k] = {
            "baseline_metrics": [],
            "reranked_metrics": [],
            "candidate_recalls": [],
            "hybrid_times": [],
            "reranker_times": [],
            "total_times": [],
        }

    # --------------------------------------------------
    # 2. WARM-UP
    # --------------------------------------------------

    print("\n" + "=" * 75)
    print("WARMING UP RETRIEVAL AND CROSS-ENCODER")
    print("=" * 75)

    for candidate_k in CANDIDATE_K_VALUES:

        warmup_candidates = hybrid_search(
            "What is RAG?",
            model,
            index,
            documents,
            top_k=TOP_K,
            candidate_k=candidate_k,
            rrf_k=60,
            min_semantic_score=0.30,
            semantic_weight=3.0,
            keyword_weight=1.0,
            return_all_candidates=True,
        )

        rerank_documents(
            "What is RAG?",
            warmup_candidates,
            top_k=TOP_K,
        )

        print(f"Warm-up complete for candidate_k={candidate_k}")

    # --------------------------------------------------
    # 3. RUN EXPERIMENTS
    # --------------------------------------------------

    for candidate_k in CANDIDATE_K_VALUES:

        print("\n\n" + "#" * 75)
        print(f"EXPERIMENT: candidate_k = {candidate_k}")
        print("#" * 75)

        results = experiment_results[candidate_k]

        for item in evaluation_dataset:

            question = item["question"]
            relevant_ids = set(item["relevant_ids"])

            print("\n" + "-" * 75)
            print("QUESTION:", question)
            print("RELEVANT IDS:", sorted(relevant_ids))

            query_hybrid_times = []
            query_reranker_times = []
            query_total_times = []

            candidates = None
            baseline_results = None
            reranked_results = None

            # --------------------------------------------------
            # 4. REPEATED MEASUREMENTS
            # --------------------------------------------------

            for repetition in range(REPETITIONS):

                # Stage 1: Hybrid retrieval.

                start = time.perf_counter()

                candidates = hybrid_search(
                    question,
                    model,
                    index,
                    documents,
                    top_k=TOP_K,
                    candidate_k=candidate_k,
                    rrf_k=60,
                    min_semantic_score=0.30,
                    semantic_weight=3.0,
                    keyword_weight=1.0,
                    return_all_candidates=True,
                )

                hybrid_time = time.perf_counter() - start

                # Baseline: RRF ranking.

                baseline_results = sorted(
                    candidates,
                    key=lambda result: result["rrf_score"],
                    reverse=True,
                )[:TOP_K]

                # Stage 2: Cross-Encoder reranking.

                start = time.perf_counter()

                reranked_results = rerank_documents(
                    question,
                    candidates,
                    top_k=TOP_K,
                )

                reranker_time = time.perf_counter() - start

                query_hybrid_times.append(hybrid_time)
                query_reranker_times.append(reranker_time)

                query_total_times.append(hybrid_time + reranker_time)

            # --------------------------------------------------
            # 5. CANDIDATE COVERAGE
            # --------------------------------------------------

            candidate_ids = [result["chunk_id"] for result in candidates]

            candidate_id_set = set(candidate_ids)

            covered_relevant_ids = candidate_id_set & relevant_ids

            missing_relevant_ids = relevant_ids - candidate_id_set

            candidate_recall = len(covered_relevant_ids) / len(relevant_ids)

            results["candidate_recalls"].append(candidate_recall)

            print("\n--- CANDIDATE COVERAGE ---")

            print("Total unique candidates:", len(candidate_ids))
            print("Candidate IDs:", candidate_ids)

            print(
                "Relevant candidates found:",
                sorted(covered_relevant_ids),
            )

            print(
                "Missing relevant candidates:",
                sorted(missing_relevant_ids),
            )

            print(f"Candidate Recall: {candidate_recall:.4f}")

            # --------------------------------------------------
            # 6. FINAL RANKING METRICS
            # --------------------------------------------------

            baseline_ids = [result["chunk_id"] for result in baseline_results]

            reranked_ids = [result["chunk_id"] for result in reranked_results]

            baseline_score = evaluate_ranking(
                baseline_ids,
                relevant_ids,
                TOP_K,
            )

            reranked_score = evaluate_ranking(
                reranked_ids,
                relevant_ids,
                TOP_K,
            )

            results["baseline_metrics"].append(baseline_score)

            results["reranked_metrics"].append(reranked_score)

            print("\n--- BASELINE RRF ---")
            print("Retrieved IDs:", baseline_ids)
            print("Metrics:", baseline_score)

            print("\n--- CROSS-ENCODER ---")
            print("Retrieved IDs:", reranked_ids)
            print("Metrics:", reranked_score)

            # --------------------------------------------------
            # 7. LATENCY
            # --------------------------------------------------

            median_hybrid = statistics.median(query_hybrid_times)

            median_reranker = statistics.median(query_reranker_times)

            median_total = statistics.median(query_total_times)

            results["hybrid_times"].append(median_hybrid)

            results["reranker_times"].append(median_reranker)

            results["total_times"].append(median_total)

            print("\n--- MEDIAN LATENCY ---")

            print(f"Hybrid retrieval: " f"{median_hybrid * 1000:.2f} ms")

            print(f"Cross-Encoder: " f"{median_reranker * 1000:.2f} ms")

            print(f"Total pipeline: " f"{median_total * 1000:.2f} ms")

    # --------------------------------------------------
    # 8. AGGREGATE METRICS
    # --------------------------------------------------

    def average_metric(metric_results, metric):

        return statistics.mean(result[metric] for result in metric_results)

    # --------------------------------------------------
    # 9. FINAL COMPARISON
    # --------------------------------------------------

    print("\n\n" + "=" * 85)
    print("FINAL CANDIDATE POOL COMPARISON")
    print("=" * 85)

    for candidate_k in CANDIDATE_K_VALUES:

        results = experiment_results[candidate_k]

        candidate_recall = statistics.mean(results["candidate_recalls"])

        baseline_mrr = average_metric(
            results["baseline_metrics"],
            "mrr",
        )

        reranked_mrr = average_metric(
            results["reranked_metrics"],
            "mrr",
        )

        baseline_precision = average_metric(
            results["baseline_metrics"],
            "precision",
        )

        reranked_precision = average_metric(
            results["reranked_metrics"],
            "precision",
        )

        baseline_recall = average_metric(
            results["baseline_metrics"],
            "recall",
        )

        reranked_recall = average_metric(
            results["reranked_metrics"],
            "recall",
        )

        median_hybrid = statistics.median(results["hybrid_times"])

        median_reranker = statistics.median(results["reranker_times"])

        median_total = statistics.median(results["total_times"])

        print("\n" + "-" * 85)
        print(f"CANDIDATE K = {candidate_k}")
        print("-" * 85)

        print(f"Candidate Recall: {candidate_recall:.4f}")

        print("\nBASELINE RRF")

        print(f"MRR@{TOP_K}: {baseline_mrr:.4f}")
        print(f"Precision@{TOP_K}: {baseline_precision:.4f}")
        print(f"Recall@{TOP_K}: {baseline_recall:.4f}")

        print("\nCROSS-ENCODER")

        print(f"MRR@{TOP_K}: {reranked_mrr:.4f}")
        print(f"Precision@{TOP_K}: {reranked_precision:.4f}")
        print(f"Recall@{TOP_K}: {reranked_recall:.4f}")

        print("\nMEDIAN LATENCY")

        print(f"Hybrid: {median_hybrid * 1000:.2f} ms")
        print(f"Reranker: {median_reranker * 1000:.2f} ms")
        print(f"Total: {median_total * 1000:.2f} ms")

    # --------------------------------------------------
    # 10. COMPARISON TABLE
    # --------------------------------------------------

    print("\n\n" + "=" * 110)
    print("EXPERIMENT COMPARISON TABLE")
    print("=" * 110)

    print(
        f"{'K':<8}"
        f"{'Cand Recall':<15}"
        f"{'Base MRR':<13}"
        f"{'Rerank MRR':<14}"
        f"{'Base Recall':<14}"
        f"{'Rerank Recall':<16}"
        f"{'Total ms':<12}"
    )

    print("-" * 110)

    for candidate_k in CANDIDATE_K_VALUES:

        results = experiment_results[candidate_k]

        print(
            f"{candidate_k:<8}"
            f"{statistics.mean(results['candidate_recalls']):<15.4f}"
            f"{average_metric(results['baseline_metrics'], 'mrr'):<13.4f}"
            f"{average_metric(results['reranked_metrics'], 'mrr'):<14.4f}"
            f"{average_metric(results['baseline_metrics'], 'recall'):<14.4f}"
            f"{average_metric(results['reranked_metrics'], 'recall'):<16.4f}"
            f"{statistics.median(results['total_times']) * 1000:<12.2f}"
        )

    print("\nEvaluation completed.")


# --------------------------------------------------
# 11. SCRIPT ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    run_retrieval_evaluation()
