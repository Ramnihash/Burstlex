from optimization.event_optimizer import (
    drop_noise_events,
    merge_duplicate_events,
    optimize_events,
    stem_key,
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
