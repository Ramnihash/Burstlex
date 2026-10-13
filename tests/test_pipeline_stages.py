from pathlib import Path

import pandas as pd

from pipeline import run_pipeline

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "tweets.csv"

EXPECTED = [
    "Lexer", "Segmentation", "Segment index", "Burst detection",
    "Adaptive symbol table", "Clustering", "Summarization",
    "Event optimizer", "Novelty", "Evaluation",
]


def test_every_stage_is_reported_in_order():
    result = run_pipeline(pd.read_csv(SAMPLE), min_frequency=2)
    assert [stage["name"] for stage in result["stages"]] == EXPECTED


def test_on_stage_callback_receives_each_stage():
    seen = []
    run_pipeline(pd.read_csv(SAMPLE), min_frequency=2, on_stage=seen.append)
    assert [stage["name"] for stage in seen] == EXPECTED
    assert all(stage["seconds"] >= 0 for stage in seen)
