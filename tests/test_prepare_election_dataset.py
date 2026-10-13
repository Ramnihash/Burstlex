import pandas as pd

from data.prepare_election_dataset import prepare_election_dataset, prepare_many


def write_sample(tmp_path):
    user = (
        "{'id': 942869257108455424, 'username': 'a', "
        "'followersCount': 152, 'friendsCount': 246}"
    )
    rows = [
        [1, "debate tonight", 1719500000.0, "en", False, 3, 10.0, user],
        [2, "retweet text", 1719500100.0, "en", True, 0, 0.0, user],
        [3, "bonjour", 1719500200.0, "fr", False, 0, 0.0, user],
        [1, "debate tonight", 1719500000.0, "en", False, 3, 10.0, user],
        [4, "another post", 1719510000.0, "en", False, None, None, user],
    ]
    columns = [
        "id", "text", "epoch", "lang", "retweetedTweet",
        "retweetCount", "likeCount", "user",
    ]
    path = tmp_path / "sample.csv.gz"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return str(path)


def test_filters_language_retweets_and_duplicates(tmp_path):
    prepared = prepare_election_dataset(write_sample(tmp_path))

    assert prepared["tweet_id"].tolist() == [1, 4]


def test_output_columns(tmp_path):
    prepared = prepare_election_dataset(write_sample(tmp_path))

    assert list(prepared.columns) == [
        "tweet_id", "timestamp", "text", "retweet_count",
        "like_count", "user_id", "followers",
    ]


def test_user_fields_and_missing_counts(tmp_path):
    prepared = prepare_election_dataset(write_sample(tmp_path))

    assert prepared["followers"].tolist() == [152, 152]
    assert prepared["user_id"].tolist() == ["942869257108455424"] * 2
    assert prepared["retweet_count"].tolist() == [3, 0]
    assert prepared["like_count"].tolist() == [10, 0]


def test_max_rows(tmp_path):
    prepared = prepare_election_dataset(write_sample(tmp_path), max_rows=1)

    assert len(prepared) == 1


def test_timestamp_is_floored_to_the_hour(tmp_path):
    prepared = prepare_election_dataset(write_sample(tmp_path))

    assert prepared["timestamp"].tolist() == [
        "2024-06-27T14:00:00",
        "2024-06-27T17:00:00",
    ]


def test_prepare_many_combines_files_without_duplicates(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    paths = [write_sample(first), write_sample(second)]

    combined = prepare_many(paths)

    assert combined["tweet_id"].tolist() == [1, 4]
    assert combined["timestamp"].is_monotonic_increasing
