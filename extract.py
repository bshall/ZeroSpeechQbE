import argparse
from pathlib import Path
import librosa
import json
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm


def extract_features(args):
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=args.num_workers)
    futures = []
    with open(args.file_list) as file:
        documents = json.load(file)
        for document in documents:
            wav_path = in_dir / document["in-path"]
            out_path = out_dir / document["out-path"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            futures.append(executor.submit(partial(process_wav, wav_path, out_path, args.sample_rate,
                                                   args.hop_length, args.win_length, document["offset"],
                                                   document["duration"])))

    results = [future.result() for future in tqdm(futures)]

    frames = sum(results)
    frame_shift_ms = args.hop_length / args.sample_rate
    hours = frames * frame_shift_ms / 3600
    print("Wrote {} utterances, {} frames ({:.2f} hours)".format(len(results), frames, hours))


def process_wav(wav_path, out_path, sample_rate, hop_length, win_length, offset=0, duration=None):
    wav, _ = librosa.load(wav_path.with_suffix(".wav"), sr=sample_rate, offset=offset, duration=duration)
    wav = wav / np.abs(wav).max() * 0.999
    wav = librosa.effects.preemphasis(wav)

    mfccs = librosa.feature.mfcc(wav, sr=sample_rate, n_mfcc=13, hop_length=hop_length,
                                 win_length=win_length, fmin=50)
    delta = librosa.feature.delta(mfccs, width=5, order=1)
    ddelta = librosa.feature.delta(mfccs, width=5, order=2)
    features = np.concatenate((mfccs, delta, ddelta))
    features = (features - features.mean(axis=-1, keepdims=True)) / features.std(axis=-1, keepdims=True)

    np.save(out_path.with_suffix(".npy"), features)
    return features.shape[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, help="path to dataset directory")
    parser.add_argument("--out-dir", type=str, help="path to output directory")
    parser.add_argument("--file-list", type=str, help="path to .yaml file listing the files to be processed")
    parser.add_argument("--num-workers", type=int, default=cpu_count())
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-length", type=int, default=200)
    parser.add_argument("--win-length", type=int, default=800)
    args = parser.parse_args()
    extract_features(args)
