import argparse
import numpy as np
import json
from pathlib import Path


def average_precision_at_n(relevant, returned, n):
    relevant_per_k = np.array([1 if document in relevant else 0 for document in returned[:n]])
    true_positives_per_k = np.cumsum(relevant_per_k)
    precision_per_k = true_positives_per_k / np.arange(1, n + 1)
    relevant_precision_per_k = precision_per_k * relevant_per_k
    return np.sum(relevant_precision_per_k) / min(len(relevant), n)


def precision_at_k(relevant, returned, k):
    return sum([1 if document in relevant else 0 for document in returned[:k]]) / k


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=str)
    args = parser.parse_args()

    with open("query.json") as file:
        queries = json.load(file)

    with open("search.json") as file:
        searches = json.load(file)

    queries_by_word = dict()
    for query in queries:
        queries_by_word.setdefault(
            query["words"][0], []
        ).append(query["out-path"])

    evaluation_dir = Path(args.evaluation_dir)
    results = {
        word: dict.fromkeys(paths) for word, paths in queries_by_word.items()
    }
    for word, paths in queries_by_word.items():
        relevant = [
            search["out-path"] for search in searches if word in search["words"]
        ]
        for path in paths:
            score_path = evaluation_dir / path
            with open(score_path.with_suffix(".json")) as file:
                returned = json.load(file)
            returned = sorted(returned, key=returned.get, reverse=True)

            p_at_n = precision_at_k(relevant, returned, len(relevant))
            avg_p_at_n = average_precision_at_n(relevant, returned, len(relevant))

            results[word][path] = {
                "precision_at_n": p_at_n,
                "average_precision_at_n": avg_p_at_n
            }

    with open(evaluation_dir / "results.json", "w") as file:
        json.dump(results, file)

    with open(evaluation_dir / "results.json", "w")

    precision_at_n_per_word = {
        word: np.mean([result["precision_at_n"] for result in queries.values()])
        for word, queries in results.items()
    }

