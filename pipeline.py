from collections import Counter

import pandas as pd

from burst_detection.bursty_segments import find_bursty_segments
from clustering.event_cluster import cluster_segments
from evaluation import count_event_tweets, evaluate_events
from indexing.segment_index import build_segment_index
from novelty.novelty_detector import analyze_event_novelty
from optimization.event_optimizer import optimize_events
from preprocessing.lexer import lex_tweet
from segmentation.adaptive_symbol_table import AdaptiveSymbolTable
from segmentation.segmenter import segment_tweet
from summarization.event_summary import summarize_event


def dataset_end_time(df):
    try:
        return pd.to_datetime(df["timestamp"]).max().to_pydatetime()
    except (ValueError, TypeError, KeyError):
        return None


def run_pipeline(df, min_frequency=2, historical_mode=False,
                 max_candidates=2000):
    tweet_segments = []
    segment_counts = Counter()
    lexical_errors = Counter()

    for row in df.itertuples(index=False):
        lexed = lex_tweet(row.text)
        lexical_errors.update(lexed.errors)
        segments = segment_tweet(lexed.text)
        segment_counts.update(segments)
        tweet_segments.append((row.tweet_id, row.timestamp, segments))

    segment_index = build_segment_index(tweet_segments)
    bursty = find_bursty_segments(
        segment_index,
        min_frequency=2 if historical_mode else min_frequency
    )

    if historical_mode and not bursty:
        detection_mode = "Historical Frequency Analysis"
        candidates = {
            segment: {
                "frequency": frequency,
                "score": float(frequency),
                "hashtag_weight": 1
            }
            for segment, frequency in segment_counts.items()
            if frequency >= min_frequency and not segment.startswith("#")
        }
        score_label = "Frequency Score"
    else:
        detection_mode = "Temporal Burst Analysis"
        candidates = {
            segment: details
            for segment, details in bursty.items()
            if not segment.startswith("#")
        }
        score_label = "Burst Score"

    candidates = dict(
        sorted(
            candidates.items(),
            key=lambda item: item[1]["score"],
            reverse=True
        )[:max_candidates]
    )

    segments = list(candidates.keys())
    symbol_table = AdaptiveSymbolTable()

    if not historical_mode:
        symbol_table.learn(segments, segment_index, tweet_segments)

    known_segments = symbol_table.filter_segments(segments)

    if known_segments:
        segments = known_segments

    events = []

    for cluster in cluster_segments(segments):
        valid_segments = [
            segment for segment in cluster if segment in candidates
        ]

        if not valid_segments:
            continue

        events.append({
            "summary": summarize_event(valid_segments),
            "frequency": count_event_tweets(valid_segments, segment_index),
            "score": max(
                candidates[segment]["score"] for segment in valid_segments
            ),
            "segments": valid_segments
        })

    events.sort(
        key=lambda event: (event["frequency"], event["score"]),
        reverse=True
    )

    optimizer_info = None

    if not historical_mode:
        events, optimizer_info = optimize_events(
            events, segment_index, tweet_segments
        )

    novelty = None

    if not historical_mode:
        novelty = analyze_event_novelty(
            df.to_dict("records"),
            {event["summary"]: event["segments"] for event in events},
            now=dataset_end_time(df)
        )

    evaluation = evaluate_events(
        df,
        [
            {
                "frequency": event["frequency"],
                "burst_score": 0 if historical_mode else event["score"]
            }
            for event in events
        ]
    )

    return {
        "events": events,
        "evaluation": evaluation,
        "novelty": novelty,
        "optimizer_info": optimizer_info,
        "lexical_errors": lexical_errors,
        "symbol_table": symbol_table,
        "detection_mode": detection_mode,
        "score_label": score_label,
        "historical_mode": historical_mode
    }
