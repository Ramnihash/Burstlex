import os
import random
import re
import sys

import pandas as pd

DEFAULT_REPO = "icedwind/x_dataset_44100"
STOPWORDS = {
    "the", "and", "is", "to", "of", "in", "for", "you", "that", "it",
    "this", "with", "are", "on", "be", "have", "was", "not", "they", "we",
}
URL_PATTERN = re.compile(r"https?://\S+")


def is_english(text):
    text = URL_PATTERN.sub("", str(text))
    if len(text) < 20:
        return False

    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    if ascii_chars / len(text) < 0.95:
        return False

    words = set(re.findall(r"[a-z']+", text.lower()))
    return len(words & STOPWORDS) >= 1


def prepare_frame(df):
    columns = {name.lower(): name for name in df.columns}
    text_col = columns["text"]
    time_col = columns["datetime"]
    user_col = columns.get("username_encoded")

    frame = pd.DataFrame({
        "text": df[text_col].astype(str),
        "timestamp": pd.to_datetime(df[time_col], errors="coerce", utc=True),
        "user_id": df[user_col].astype(str) if user_col else "",
    })

    frame = frame.dropna(subset=["timestamp"])
    frame = frame[~frame["text"].str.startswith("RT @")]
    frame = frame[frame["text"].map(is_english)]
    frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    return frame.drop_duplicates(subset="text").reset_index(drop=True)


def finalize(frame, sample_size, seed):
    if len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=seed)

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame.insert(0, "tweet_id", range(1, len(frame) + 1))

    return frame[["tweet_id", "text", "timestamp", "user_id"]]


def read_file(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)

    return pd.read_csv(path)


def collect(repo, sample_size, pool_factor, seed):
    from huggingface_hub import HfApi, hf_hub_download

    files = [
        name for name in HfApi().list_repo_files(repo, repo_type="dataset")
        if name.endswith(".parquet") or name.endswith(".csv")
    ]
    random.Random(seed).shuffle(files)
    print("Data files in repo:", len(files))

    parts = []
    pool = 0
    target = sample_size * pool_factor

    for number, name in enumerate(files, 1):
        path = hf_hub_download(repo, name, repo_type="dataset")
        part = prepare_frame(read_file(path))
        parts.append(part)
        pool += len(part)
        print(f"File {number}/{len(files)}: kept {len(part)}, pool {pool}")

        if pool >= target:
            break

    combined = pd.concat(parts, ignore_index=True)
    return combined.drop_duplicates(subset="text").reset_index(drop=True)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: py data\\prepare_x_dataset.py <output_csv> "
            "[<rows>] [<repo>]"
        )
        return

    output_path = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    repo = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_REPO

    pool = collect(repo, sample_size, 2, 42)
    result = finalize(pool, sample_size, 42)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result.to_csv(output_path, index=False)

    print("Tweets saved:", len(result))
    print("Date range:", result["timestamp"].min(), "to", result["timestamp"].max())
    print("Tweets per day:")
    print(result["timestamp"].str[:10].value_counts().sort_index().to_string())
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
