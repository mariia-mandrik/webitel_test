import json

from app.rag.retriever import search


TOP_K = 3


def load_dataset():
    with open(
        "tests/golden_dataset.json",
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def calculate_retrieval_metrics(dataset):
    hits = 0
    total_expected_sources = 0
    found_expected_sources = 0
    retrieved_relevant = 0
    total_retrieved = 0

    for case in dataset:
        question = case["question"]
        expected_sources = set(case["expected_sources"])

        results = search(
            question,
            limit=TOP_K,
        )
        print("\n--------------------------------")
        print(f"Question: {question}")
        print(f"Expected sources: {expected_sources}")
        print(f"Retrieved results: {results}")

        retrieved_sources = [
            result["source_id"]
            for result in results
        ]

        retrieved_sources_set = set(retrieved_sources)

        # Hit@K
        if expected_sources & retrieved_sources_set:
            hits += 1

        # Recall@K
        found = expected_sources & retrieved_sources_set

        found_expected_sources += len(found)
        total_expected_sources += len(expected_sources)

        # Precision@K
        relevant = retrieved_sources_set & expected_sources

        retrieved_relevant += len(relevant)
        total_retrieved += len(retrieved_sources)

    hit_rate = hits / len(dataset) if dataset else 0

    recall = (
        found_expected_sources / total_expected_sources
        if total_expected_sources
        else 0
    )

    precision = (
        retrieved_relevant / total_retrieved
        if total_retrieved
        else 0
    )

    return {
        "hit@3": hit_rate,
        "recall@3": recall,
        "precision@3": precision,
    }


def main():
    dataset = load_dataset()

    metrics = calculate_retrieval_metrics(dataset)

    print("\n========== RETRIEVAL ==========")

    for name, value in metrics.items():
        print(f"{name}: {value:.2%}")


if __name__ == "__main__":
    main()