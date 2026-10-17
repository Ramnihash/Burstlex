from pathlib import Path

import pandas as pd

from pipeline import run_pipeline,compact_days

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "tweets.csv"


def test_pipeline_returns_expected_sections():
    df = pd.read_csv(SAMPLE)
    result = run_pipeline(df, min_frequency=2)

    for key in (
        "events", "evaluation", "novelty", "optimizer_info","event_rows",
        "lexical_errors", "symbol_table", "detection_mode", "score_label"
    ):
        assert key in result

    assert result["evaluation"]["total_tweets"] == len(df)
    assert result["detection_mode"] == "Temporal Burst Analysis"


def test_pipeline_historical_mode_skips_optimizer_and_novelty():
    df = pd.DataFrame({
        "tweet_id": [1, 2, 3, 4],
        "timestamp": ["historical_unknown"] * 4,
        "text": ["boston marathon bombing news"] * 4,
    })
    result = run_pipeline(df, min_frequency=2, historical_mode=True)

    assert result["optimizer_info"] is None
    assert result["novelty"] is None
    assert result["detection_mode"] == "Historical Frequency Analysis"


def test_compact_days_closes_gaps_and_keeps_time_of_day():
    df = pd.DataFrame({
        "tweet_id": [1, 2, 3],
        "timestamp": [
            "2025-01-27 10:00:00",
            "2025-01-28 23:30:00",
            "2025-04-06 08:15:00",
        ],
    })
    out = compact_days(df)
    assert list(out["timestamp"]) == [
    "2025-01-27 10:00:00",
    "2025-01-28 23:30:00",
    "2025-01-29 08:15:00",
    ]
    assert list(df["timestamp"])[2] == "2025-04-06 08:15:00"


def test_compact_days_leaves_consecutive_days_unchanged():
    df = pd.DataFrame({
        "tweet_id": [1, 2],
        "timestamp": ["2024-07-01 10:00:00", "2024-07-02 11:00:00"],
    })
    assert list(compact_days(df)["timestamp"]) == list(df["timestamp"])


def test_compact_days_returns_input_when_timestamps_unreadable():
    df = pd.DataFrame({"tweet_id": [1], "timestamp": ["unknown"]})
    assert compact_days(df) is df

