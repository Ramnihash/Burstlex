import sys

import pandas as pd

USE_COLUMNS = [
    "id",
    "text",
    "epoch",
    "lang",
    "retweetedTweet",
    "retweetCount",
    "likeCount",
    "user",
]


def prepare_election_dataset(input_path, max_rows=None):
    df = pd.read_csv(input_path, usecols=USE_COLUMNS, nrows=max_rows)

    df = df[df["lang"] == "en"]
    df = df[df["retweetedTweet"].astype(str).str.lower() != "true"]
    df = df.dropna(subset=["id", "text", "epoch"])
    df = df.drop_duplicates(subset="id")

    followers = df["user"].astype(str).str.extract(r"'followersCount': (\d+)")[0]
    user_ids = df["user"].astype(str).str.extract(r"'id': (\d+)")[0]

    hours = pd.to_datetime(df["epoch"], unit="s", utc=True).dt.floor("h")

    prepared = pd.DataFrame({
        "tweet_id": df["id"].astype("int64"),
        "timestamp": hours.dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "text": df["text"],
        "retweet_count": df["retweetCount"].fillna(0).astype(int),
        "like_count": df["likeCount"].fillna(0).astype(int),
        "user_id": user_ids,
        "followers": followers.fillna(0).astype(int),
    })

    return prepared.reset_index(drop=True)


def prepare_many(input_paths, max_rows_per_file=None):
    parts = [
        prepare_election_dataset(path, max_rows_per_file)
        for path in input_paths
    ]
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset="tweet_id")

    return combined.sort_values("timestamp").reset_index(drop=True)


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: py data\\prepare_election_dataset.py "
            "<output_csv> <rows_per_file> <input_csv_gz> [<input_csv_gz> ...]"
        )
        return

    output_path = sys.argv[1]
    max_rows = int(sys.argv[2])
    input_paths = sys.argv[3:]

    prepared = prepare_many(input_paths, max_rows)
    prepared.to_csv(output_path, index=False)

    print("Files combined:", len(input_paths))
    print("Tweets saved:", len(prepared))
    print("Date range:", prepared["timestamp"].min(), "to", prepared["timestamp"].max())
    print("Distinct days:", prepared["timestamp"].str[:10].nunique())
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
