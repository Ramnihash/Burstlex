from segmentation.adaptive_symbol_table import AdaptiveSymbolTable


def make_index(mapping):
    return {
        segment: [{"tweet_id": t, "timestamp": "x", "is_hashtag": False} for t in ids]
        for segment, ids in mapping.items()
    }


def make_tweets():
    tweets = []
    for i in range(100):
        day = "2024-01-01" if i < 30 else "2024-01-02"
        tweets.append((i, day + "T10:00:00", []))
    return tweets


TERMS = {"known title"}


def test_wikipedia_phrase_is_contained():
    table = AdaptiveSymbolTable(TERMS)
    assert table.contains("known title")
    assert not table.contains("unknown phrase")


def test_bursty_unknown_phrase_is_learned():
    index = make_index({"fresh scandal": list(range(0, 28))})
    table = AdaptiveSymbolTable(TERMS)
    learned = table.learn(["fresh scandal"], index, make_tweets(), min_frequency=25)
    assert learned == {"fresh scandal"}
    assert table.contains("fresh scandal")


def test_flat_phrase_is_not_learned():
    index = make_index({"every day": list(range(0, 100, 2))})
    table = AdaptiveSymbolTable(TERMS)
    assert table.learn(["every day"], index, make_tweets(), min_frequency=25) == set()


def test_rare_phrase_is_not_learned():
    index = make_index({"tiny burst": [1, 2, 3]})
    table = AdaptiveSymbolTable(TERMS)
    assert table.learn(["tiny burst"], index, make_tweets(), min_frequency=25) == set()


def test_non_ascii_phrase_is_not_learned():
    index = make_index({"café crowd": list(range(0, 28))})
    table = AdaptiveSymbolTable(TERMS)
    assert table.learn(["café crowd"], index, make_tweets(), min_frequency=25) == set()


def test_single_day_learns_nothing():
    tweets = [(i, "2024-01-01T10:00:00", []) for i in range(100)]
    index = make_index({"fresh scandal": list(range(0, 28))})
    table = AdaptiveSymbolTable(TERMS)
    assert table.learn(["fresh scandal"], index, tweets) == set()


def test_filter_segments_keeps_known_and_learned():
    index = make_index({"fresh scandal": list(range(0, 28))})
    table = AdaptiveSymbolTable(TERMS)
    table.learn(["fresh scandal"], index, make_tweets(), min_frequency=25)
    assert table.filter_segments(["known title", "fresh scandal", "other thing"]) == [
        "known title",
        "fresh scandal",
    ]
