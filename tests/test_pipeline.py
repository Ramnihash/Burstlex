from pathlib import Path

import pandas as pd

from pipeline import run_pipeline

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
