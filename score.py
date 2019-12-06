import argparse
from pathlib import Path
import librosa
import json
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm


def score_features(args):
    with open("query.json") as file:
        queries = json.load(file)

    with open("search.json") as file:
        searches = json.load(file)

    feature_dir, evaluation_dir = Path(args.feature_dir), Path(args.evaluation_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=args.num_workers)

    for query in tqdm(queries):
        futures = []
        for search in searches:
            query_path = feature_dir / query["out-path"]
            search_path = feature_dir / search["out-path"]
            futures.append((search["out-path"], executor.submit(partial(dtw_score, query_path, search_path))))

        results = {search: future.result() for search, future in tqdm(futures, leave=False)}
        out_path = evaluation_dir / query["out-path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path.with_suffix(".json"), "w") as file:
            json.dump(results, file, indent=4, sort_keys=True)


def dtw_score(query_path, search_path):
    query = np.load(query_path.with_suffix(".npy"))
    search = np.load(search_path.with_suffix(".npy"))
    D, wp = librosa.sequence.dtw(query, search, subseq=True, metric="euclidean")
    score = -np.min(D[-1, :] / wp.shape[0])
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=str, help="directory containing features to evaluate")
    parser.add_argument("--evaluation-dir", type=str)
    parser.add_argument("--num-workers", type=int, default=cpu_count())
    args = parser.parse_args()
    score_features(args)
