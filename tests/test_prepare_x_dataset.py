import pandas as pd

from data.prepare_x_dataset import finalize, is_english, prepare_frame


def sample_frame():
    rows = [
        ["the government announced a new budget today #news", "2025-01-22T10:00:00Z", "u1"],
        ["RT @someone: the government announced a new budget", "2025-01-22T10:01:00Z", "u2"],
        ["今日はとても良い天気ですね、散歩に行きましょう", "2025-01-22T10:02:00Z", "u3"],
        ["the government announced a new budget today #news", "2025-01-22T10:03:00Z", "u4"],
        ["short", "2025-01-22T10:04:00Z", "u5"],
        ["markets fell sharply and the dollar is weaker now", "bad date", "u6"],
        ["wildfire smoke is spreading across the city this evening", "2025-01-23T08:30:00Z", "u7"],
    ]
    return pd.DataFrame(rows, columns=["text", "datetime", "username_encoded"])


def test_is_english():
    assert is_english("the government announced a new budget today")
    assert not is_english("short")
    assert not is_english("今日はとても良い天気ですね、散歩に行きましょう")
    assert not is_english("https://t.co/abcdefghijk the")


def test_prepare_frame_filters_rows():
    frame = prepare_frame(sample_frame())

    assert frame["text"].tolist() == [
        "the government announced a new budget today #news",
        "wildfire smoke is spreading across the city this evening",
    ]
    assert frame["timestamp"].tolist() == [
        "2025-01-22T10:00:00",
        "2025-01-23T08:30:00",
    ]
    assert frame["user_id"].tolist() == ["u1", "u7"]


def test_finalize_samples_and_numbers_rows():
    frame = prepare_frame(sample_frame())
    result = finalize(frame, 1, 42)

    assert list(result.columns) == ["tweet_id", "text", "timestamp", "user_id"]
    assert result["tweet_id"].tolist() == [1]

    full = finalize(frame, 10, 42)
    assert full["tweet_id"].tolist() == [1, 2]
    assert full["timestamp"].is_monotonic_increasing
