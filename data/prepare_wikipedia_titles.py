import gzip
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation.segmenter import segment_tweet


def collect_candidates(dataset_path):
    df = pd.read_csv(dataset_path)
    candidates = set()

    for text in df["text"]:
        candidates.update(
            segment
            for segment in segment_tweet(text)
            if not segment.startswith("#")
        )

    return candidates


def open_dump(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")

    return open(path, encoding="utf-8", errors="ignore")


def extract_titles(dump_path, candidates):
    matched = set()

    with open_dump(dump_path) as dump:
        for line in dump:
            title = line.strip().replace("_", " ").lower()

            if title in candidates:
                matched.add(title)

    return matched


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: py data\\prepare_wikipedia_titles.py "
            "<titles_dump> <dataset_csv> <output_txt>"
        )
        return

    dump_path, dataset_path, output_path = sys.argv[1:4]

    candidates = collect_candidates(dataset_path)
    matched = extract_titles(dump_path, candidates)

    with open(output_path, "w", encoding="utf-8") as output:
        for title in sorted(matched):
            output.write(title + "\n")

    print("Candidate segments:", len(candidates))
    print("Matched Wikipedia titles:", len(matched))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
