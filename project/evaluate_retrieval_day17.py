
from faiss_text import (
    semantic_search,
    hybrid_search,
    model,
    index,
    documents,
)


# ============================================================
# 1. EVALUATION DATASET
# ============================================================

EVALUATION_DATASET = [
    {"question": "What is Python used for?", "relevant_ids": {1}},
    {"question": "What is Python and what makes it readable?", "relevant_ids": {0}},
    {"question": "What is semantic search?", "relevant_ids": {28}},
    {"question": "What is an embedding?", "relevant_ids": {27}},
    {"question": "What is RAG and how does it work?", "relevant_ids": {23, 24, 25}},
    {"question": "Why is RAG useful?", "relevant_ids": {26}},
    {"question": "What is chunking and why is it used?", "relevant_ids": {40}},
    {"question": "What does FAISS do?", "relevant_ids": {30, 31}},
    {"question": "What does top-k retrieval mean?", "relevant_ids": {35}},
    {"question": "What is a similarity threshold?", "relevant_ids": {36}},
    {"question": "Which embedding model does our project use?", "relevant_ids": {33}},
    {"question": "What is Ollama?", "relevant_ids": {19}},
    {"question": "What is FastAPI used for?", "relevant_ids": {14}},
    {"question": "What is a large language model?", "relevant_ids": {17}},
    {
        "question": "Which project files handle retrieval and answer generation?",
        "relevant_ids": {42},
    },
]


# ============================================================
# 2. VALIDATE THE EVALUATION DATASET
# ============================================================

def validate_evaluation_dataset():

    available_ids = {
        chunk["chunk_id"] for chunk in documents
    }

    for item in EVALUATION_DATASET:

        missing_ids = item["relevant_ids"] - available_ids

        if missing_ids:
            raise ValueError(
                f"Invalid relevant IDs for "
                f"{item['question']}: {missing_ids}"
            )

    if not EVALUATION_DATASET:
        raise ValueError("Evaluation dataset cannot be empty.")

    print("Evaluation dataset validation passed.")


# ============================================================
# 3. EVALUATE A SINGLE QUERY
# ============================================================

def evaluate_query(
    question,
    relevant_ids,
    k,
    method="semantic",
    candidate_k=10,
    rrf_k=60,
    semantic_weight=1.0,
    keyword_weight=1.0,
):

    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer.")

    if method == "semantic":

        results = semantic_search(
            question,
            model,
            index,
            documents,
            top_k=k,
            min_score=None,
        )

    elif method == "hybrid":

        results = hybrid_search(
            question,
            model,
            index,
            documents,
            top_k=k,
            candidate_k=max(candidate_k, k),
            rrf_k=rrf_k,
            min_semantic_score=None,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
        )

    else:
        raise ValueError(
            "method must be 'semantic' or 'hybrid'."
        )

    retrieved_ids = [
        result["chunk_id"] for result in results
    ]

    relevant_retrieved = set(retrieved_ids) & relevant_ids

    precision = len(relevant_retrieved) / k

    recall = (
        len(relevant_retrieved) / len(relevant_ids)
        if relevant_ids
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "retrieved_ids": retrieved_ids,
        "matched_ids": sorted(relevant_retrieved),
    }


# ============================================================
# 4. EVALUATE A RETRIEVAL METHOD ACROSS THE DATASET
# ============================================================

def evaluate_method(
    method,
    k_values=(1, 3, 5),
    candidate_k=10,
    rrf_k=60,
    semantic_weight=1.0,
    keyword_weight=1.0,
):

    all_results = {}

    for k in k_values:

        precision_scores = []
        recall_scores = []

        for item in EVALUATION_DATASET:

            result = evaluate_query(
                question=item["question"],
                relevant_ids=item["relevant_ids"],
                k=k,
                method=method,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
            )

            precision_scores.append(result["precision"])
            recall_scores.append(result["recall"])

        all_results[k] = {
            "precision": sum(precision_scores) / len(precision_scores),
            "recall": sum(recall_scores) / len(recall_scores),
        }

    return all_results


# ============================================================
# 5. PRINT SEMANTIC VS HYBRID COMPARISON
# ============================================================

def print_comparison(semantic_metrics, hybrid_metrics):

    print("\n" + "=" * 75)
    print("SEMANTIC SEARCH VS HYBRID SEARCH")
    print("=" * 75)

    print(
        f"\n{'K':<5}"
        f"{'Semantic P@K':<18}"
        f"{'Hybrid P@K':<18}"
        f"{'Semantic R@K':<18}"
        f"{'Hybrid R@K':<18}"
    )

    print("-" * 75)

    for k in (1, 3, 5):

        semantic = semantic_metrics[k]
        hybrid = hybrid_metrics[k]

        print(
            f"{k:<5}"
            f"{semantic['precision']:<18.4f}"
            f"{hybrid['precision']:<18.4f}"
            f"{semantic['recall']:<18.4f}"
            f"{hybrid['recall']:<18.4f}"
        )


# ============================================================
# 6. PRINT PER-QUESTION RESULTS
# ============================================================

def print_per_question_results(k=3):

    print("\n" + "=" * 75)
    print(f"PER-QUESTION COMPARISON AT K = {k}")
    print("=" * 75)

    for item in EVALUATION_DATASET:

        question = item["question"]
        relevant_ids = item["relevant_ids"]

        semantic_result = evaluate_query(
            question,
            relevant_ids,
            k,
            method="semantic",
        )

        hybrid_result = evaluate_query(
            question,
            relevant_ids,
            k,
            method="hybrid",
        )

        print(f"\nQuestion: {question}")
        print(f"Expected relevant IDs: {sorted(relevant_ids)}")

        print(
            f"Semantic retrieved: {semantic_result['retrieved_ids']}"
        )

        print(
            f"Semantic matched:   {semantic_result['matched_ids']}"
        )

        print(
            f"Semantic P@{k}: {semantic_result['precision']:.4f}"
            f" | R@{k}: {semantic_result['recall']:.4f}"
        )

        print(
            f"Hybrid retrieved:   {hybrid_result['retrieved_ids']}"
        )

        print(
            f"Hybrid matched:     {hybrid_result['matched_ids']}"
        )

        print(
            f"Hybrid P@{k}: {hybrid_result['precision']:.4f}"
            f" | R@{k}: {hybrid_result['recall']:.4f}"
        )


# ============================================================
# 7. TEST DIFFERENT QUERY WORDINGS
# ============================================================

def run_query_wording_experiment():

    print("\n" + "=" * 75)
    print("QUERY WORDING EXPERIMENT")
    print("=" * 75)

    test_queries = [
        "Which project files handle retrieval and answer generation?",
        "Which Python file performs semantic retrieval?",
        "Which file calls Ollama to generate answers?",
        (
            "What are the responsibilities of "
            "faiss_text.py, rag_context.py, and llm_service.py?"
        ),
    ]

    target_chunk_id = 42

    for question in test_queries:

        print(f"\nQuestion: {question}")

        for method in ("semantic", "hybrid"):

            if method == "semantic":

                results = semantic_search(
                    question,
                    model,
                    index,
                    documents,
                    top_k=5,
                    min_score=None,
                )

            else:

                results = hybrid_search(
                    question,
                    model,
                    index,
                    documents,
                    top_k=5,
                    candidate_k=10,
                    rrf_k=60,
                    min_semantic_score=None,
                )

            ranked_ids = [
                result["chunk_id"] for result in results
            ]

            if target_chunk_id in ranked_ids:

                rank = ranked_ids.index(target_chunk_id) + 1

                print(
                    f"  {method.title()} rank for chunk "
                    f"{target_chunk_id}: {rank}"
                )

            else:

                print(
                    f"  {method.title()} rank for chunk "
                    f"{target_chunk_id}: outside top 5"
                )

            print(f"  Retrieved IDs: {ranked_ids}")


# ============================================================
# 8. RRF PARAMETER TUNING EXPERIMENT
# ============================================================

def run_rrf_tuning_experiment():

    print("\n" + "=" * 75)
    print("DAY 18: RRF PARAMETER TUNING")
    print("=" * 75)

    candidate_k = 10
    rrf_values = [10, 30, 60, 100]
    k_values = (1, 3, 5)

    tuning_results = {}

    for rrf_k in rrf_values:

        print("\n" + "-" * 60)

        print(
            f"Testing candidate_k={candidate_k}, "
            f"rrf_k={rrf_k}"
        )

        print("-" * 60)

        metrics = evaluate_method(
            method="hybrid",
            k_values=k_values,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
        )

        tuning_results[rrf_k] = metrics

        for k, values in metrics.items():

            print(
                f"K={k} | "
                f"Precision={values['precision']:.4f} | "
                f"Recall={values['recall']:.4f}"
            )

    print("\n" + "=" * 75)
    print("RRF TUNING SUMMARY")
    print("=" * 75)

    print(
        f"\n{'RRF K':<10}"
        f"{'P@1':<12}{'R@1':<12}"
        f"{'P@3':<12}{'R@3':<12}"
        f"{'P@5':<12}{'R@5':<12}"
    )

    print("-" * 75)

    for rrf_k, metrics in tuning_results.items():

        print(
            f"{rrf_k:<10}"
            f"{metrics[1]['precision']:<12.4f}"
            f"{metrics[1]['recall']:<12.4f}"
            f"{metrics[3]['precision']:<12.4f}"
            f"{metrics[3]['recall']:<12.4f}"
            f"{metrics[5]['precision']:<12.4f}"
            f"{metrics[5]['recall']:<12.4f}"
        )

    return tuning_results


# ============================================================
# 9. RRF RANKING STABILITY EXPERIMENT
# ============================================================

def compare_rrf_rankings():

    print("\n" + "=" * 75)
    print("RRF RANKING STABILITY EXPERIMENT")
    print("=" * 75)

    rrf_values = [10, 30, 60, 100]
    candidate_k = 10
    top_k = 5

    baseline_rrf = 60
    changed_questions = 0

    for item in EVALUATION_DATASET:

        question = item["question"]
        rankings = {}

        for rrf_k in rrf_values:

            result = evaluate_query(
                question=question,
                relevant_ids=item["relevant_ids"],
                k=top_k,
                method="hybrid",
                candidate_k=candidate_k,
                rrf_k=rrf_k,
            )

            rankings[rrf_k] = result["retrieved_ids"]

        baseline = rankings[baseline_rrf]

        if any(
            rankings[rrf_k] != baseline
            for rrf_k in rrf_values
        ):

            changed_questions += 1

            print(f"\nQuestion: {question}")

            for rrf_k in rrf_values:

                print(
                    f"rrf_k={rrf_k:<3} "
                    f"Top-{top_k}: {rankings[rrf_k]}"
                )

    print("\n" + "-" * 75)

    print(
        f"Questions with ranking changes: "
        f"{changed_questions}/{len(EVALUATION_DATASET)}"
    )

    print("-" * 75)


# ============================================================
# 10. CANDIDATE K TUNING EXPERIMENT
# ============================================================

def run_candidate_k_experiment():

    print("\n" + "=" * 75)
    print("DAY 18: CANDIDATE K TUNING")
    print("=" * 75)

    candidate_values = [5, 10, 15, 20]
    rrf_k = 60
    k_values = (1, 3, 5)

    results_by_candidate = {}

    for candidate_k in candidate_values:

        print("\n" + "-" * 60)

        print(
            f"Testing candidate_k={candidate_k}, "
            f"rrf_k={rrf_k}"
        )

        print("-" * 60)

        metrics = evaluate_method(
            method="hybrid",
            k_values=k_values,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
        )

        results_by_candidate[candidate_k] = metrics

        for k, values in metrics.items():

            print(
                f"K={k} | "
                f"Precision={values['precision']:.4f} | "
                f"Recall={values['recall']:.4f}"
            )

    print("\n" + "=" * 75)
    print("CANDIDATE K TUNING SUMMARY")
    print("=" * 75)

    print(
        f"\n{'Candidate K':<15}"
        f"{'P@1':<12}{'R@1':<12}"
        f"{'P@3':<12}{'R@3':<12}"
        f"{'P@5':<12}{'R@5':<12}"
    )

    print("-" * 75)

    for candidate_k, metrics in results_by_candidate.items():

        print(
            f"{candidate_k:<15}"
            f"{metrics[1]['precision']:<12.4f}"
            f"{metrics[1]['recall']:<12.4f}"
            f"{metrics[3]['precision']:<12.4f}"
            f"{metrics[3]['recall']:<12.4f}"
            f"{metrics[5]['precision']:<12.4f}"
            f"{metrics[5]['recall']:<12.4f}"
        )

    return results_by_candidate


# ============================================================
# 11. DAY 20: WEIGHTED RRF EVALUATION
# ============================================================

def run_weighted_rrf_experiment():

    print("\n" + "=" * 85)
    print("DAY 20: WEIGHTED RRF EVALUATION")
    print("=" * 85)

    candidate_k = 10
    rrf_k = 60
    k_values = (1, 3, 5)

    configurations = [
        ("Semantic Only", "semantic", 1.0, 1.0),
        ("Standard RRF", "hybrid", 1.0, 1.0),
        ("Semantic 2:1", "hybrid", 2.0, 1.0),
        ("Keyword 1:2", "hybrid", 1.0, 2.0),
        ("Semantic 3:1", "hybrid", 3.0, 1.0),
    ]

    all_results = {}

    for name, method, semantic_weight, keyword_weight in configurations:

        print("\n" + "-" * 65)
        print(f"Testing: {name}")

        metrics = evaluate_method(
            method=method,
            k_values=k_values,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
        )

        all_results[name] = metrics

        for k, values in metrics.items():

            print(
                f"K={k} | "
                f"Precision={values['precision']:.4f} | "
                f"Recall={values['recall']:.4f}"
            )

    print("\n" + "=" * 85)
    print("WEIGHTED RRF COMPARISON SUMMARY")
    print("=" * 85)

    print(
        f"\n{'Configuration':<20}"
        f"{'P@1':<10}{'R@1':<10}"
        f"{'P@3':<10}{'R@3':<10}"
        f"{'P@5':<10}{'R@5':<10}"
    )

    print("-" * 85)

    for name, metrics in all_results.items():

        print(
            f"{name:<20}"
            f"{metrics[1]['precision']:<10.4f}"
            f"{metrics[1]['recall']:<10.4f}"
            f"{metrics[3]['precision']:<10.4f}"
            f"{metrics[3]['recall']:<10.4f}"
            f"{metrics[5]['precision']:<10.4f}"
            f"{metrics[5]['recall']:<10.4f}"
        )

    return all_results


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():

    # Step 1: Validate dataset.
    validate_evaluation_dataset()

    print("\n" + "=" * 75)
    print("RETRIEVAL EVALUATION")
    print("=" * 75)

    print(f"Knowledge chunks: {len(documents)}")
    print(f"FAISS vectors: {index.ntotal}")
    print(f"Evaluation questions: {len(EVALUATION_DATASET)}")

    k_values = (1, 3, 5)

    # Step 2: Semantic evaluation.
    semantic_metrics = evaluate_method(
        method="semantic",
        k_values=k_values,
    )

    # Step 3: Standard hybrid evaluation.
    hybrid_metrics = evaluate_method(
        method="hybrid",
        k_values=k_values,
        candidate_k=10,
        rrf_k=60,
    )

    # Step 4: Semantic vs hybrid comparison.
    print_comparison(
        semantic_metrics,
        hybrid_metrics,
    )

    # Step 5: Per-question comparison.
    print_per_question_results(k=3)

    # Step 6: Query wording experiment.
    run_query_wording_experiment()

    # Step 7: RRF parameter tuning.
    run_rrf_tuning_experiment()

    # Step 8: RRF ranking stability.
    compare_rrf_rankings()

    # Step 9: Candidate K tuning.
    run_candidate_k_experiment()

    # Step 10: Weighted RRF evaluation.
    run_weighted_rrf_experiment()

    print("\n" + "=" * 75)
    print("RETRIEVAL EVALUATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
