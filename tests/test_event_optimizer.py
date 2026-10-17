from optimization.event_optimizer import (
    drop_noise_events,
    merge_duplicate_events,
    optimize_events,
    stem_key,
    describe_events,
)


def make_index(mapping):
    return {
        segment: [{"tweet_id": t, "timestamp": "x", "is_hashtag": False} for t in ids]
        for segment, ids in mapping.items()
    }


def make_event(summary, segments, frequency=10):
    return {
        "summary": summary,
        "frequency": frequency,
        "score": 1.0,
        "segments": segments,
    }


def test_stem_key_matches_inflections():
    assert stem_key("dropping out") == stem_key("dropped out")


def test_merge_by_stem_key():
    index = make_index({"drop out": [1, 2], "dropped out": [3, 4]})
    events = [
        make_event("drop out", ["drop out"]),
        make_event("dropped out", ["dropped out"]),
    ]
    merged = merge_duplicate_events(events, index)
    assert len(merged) == 1
    assert merged[0]["frequency"] == 4


def test_merge_by_tweet_overlap():
    index = make_index({"alpha beta": [1, 2, 3, 4], "gamma delta": [1, 2, 3, 4]})
    events = [
        make_event("alpha beta", ["alpha beta"]),
        make_event("gamma delta", ["gamma delta"]),
    ]
    assert len(merge_duplicate_events(events, index)) == 1


def test_distinct_events_not_merged():
    index = make_index({"alpha beta": [1, 2], "gamma delta": [3, 4]})
    events = [
        make_event("alpha beta", ["alpha beta"]),
        make_event("gamma delta", ["gamma delta"]),
    ]
    assert len(merge_duplicate_events(events, index)) == 2


def test_noise_dropped_and_spike_kept():
    tweets = []
    for i in range(100):
        day = "2024-01-01" if i < 50 else "2024-01-02"
        tweets.append((i, day + " 10:00:00", []))
    flat = list(range(0, 100, 5))
    spike = list(range(0, 10))
    index = make_index({"flat phrase": flat, "spike phrase": spike})
    events = [
        make_event("flat phrase", ["flat phrase"]),
        make_event("spike phrase", ["spike phrase"]),
    ]
    kept, dropped = drop_noise_events(events, index, tweets)
    assert [e["summary"] for e in kept] == ["spike phrase"]
    assert [e["summary"] for e in dropped] == ["flat phrase"]


def test_single_day_skips_noise_pass():
    tweets = [(i, "2024-01-01 10:00:00", []) for i in range(10)]
    index = make_index({"alpha beta": [1, 2]})
    events = [make_event("alpha beta", ["alpha beta"])]
    kept, dropped = drop_noise_events(events, index, tweets)
    assert len(kept) == 1 and dropped == []


def test_optimize_events_reports_counts():
    tweets = [(i, "2024-01-01 10:00:00", []) for i in range(10)]
    index = make_index({"drop out": [1, 2], "dropped out": [3, 4]})
    events = [
        make_event("drop out", ["drop out"]),
        make_event("dropped out", ["dropped out"]),
    ]
    kept, info = optimize_events(events, index, tweets)
    assert info["before"] == 2
    assert info["after_merge"] == 1
    assert info["after_noise"] == 1


def test_describe_events_reports_days_lift_and_examples():
    index = make_index({"alpha beta": [1, 2, 3, 4], "gamma delta": [5, 6]})
    tweet_segments = [
        (1, "2025-01-01T00:00:00", []),
        (2, "2025-01-01T00:00:00", []),
        (3, "2025-01-01T00:00:00", []),
        (4, "2025-01-02T00:00:00", []),
        (5, "2025-01-02T00:00:00", []),
        (6, "2025-01-02T00:00:00", []),
    ]
    texts = {1: "first tweet", 2: "second tweet", 3: "third tweet"}
    users = {1: "a", 2: "a", 3: "b", 4: "c"}
    events = [make_event("alpha beta", ["alpha beta"], frequency=4)]
    rows = describe_events(events, index, tweet_segments, texts, users)
    assert len(rows) == 1
    row = rows[0]
    assert row["rank"] == 1
    assert row["event"] == "alpha beta"
    assert row["days"] == "2025-01-01:3; 2025-01-02:1"
    assert row["peak_day"] == "2025-01-01"
    assert row["peak_share"] == 0.75
    assert row["lift"] == 1.5
    assert row["top_user_share"] == 0.5
    assert row["example_tweets"] == "first tweet || second tweet"
    assert row["is_event"] == ""
    assert row["duplicate_of"] == ""

def test_describe_events_has_no_lift_for_a_single_day():
    index = make_index({"alpha beta": [1, 2]})
    tweet_segments = [
        (1, "unknown", []),
        (2, "unknown", []),
    ]
    events = [make_event("alpha beta", ["alpha beta"], frequency=2)]
    rows = describe_events(events, index, tweet_segments)
    assert rows[0]["lift"] == ""
    assert rows[0]["example_tweets"] == ""


def test_optimize_events_ranks_focused_events_first():
    tweets = []
    for i in range(40):
        day = "2024-01-01" if i < 20 else "2024-01-02"
        tweets.append((i, day + " 10:00:00", []))
    spread = list(range(0, 40, 2))
    focused = list(range(0, 10))
    index = make_index({"spread phrase": spread, "focused phrase": focused})
    events = [
        make_event("spread phrase", ["spread phrase"], frequency=20),
        make_event("focused phrase", ["focused phrase"], frequency=10),
    ]
    kept, info = optimize_events(events, index, tweets, min_lift=0)
    assert [e["summary"] for e in kept] == ["focused phrase", "spread phrase"]
    assert kept[0]["focus"] > kept[1]["focus"]
