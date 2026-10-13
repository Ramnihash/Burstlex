import time
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


def compact(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if value >= 10_000:
        return f"{value / 1_000:.0f}K"

    return f"{value:,}"


def dataset_end_time(df):
    try:
        return pd.to_datetime(df["timestamp"]).max().to_pydatetime()
    except (ValueError, TypeError, KeyError):
        return None


def run_pipeline(df, min_frequency=2, historical_mode=False,
                 max_candidates=2000, on_stage=None):
    stages = []
    clock = [time.perf_counter()]

    def stage(name, summary, headline, label, details=None):
        now = time.perf_counter()
        record = {
            "name": name,
            "summary": summary,
            "headline": headline,
            "label": label,
            "details": details or {},
            "seconds": now - clock[0]
        }
        clock[0] = now
        stages.append(record)

        if on_stage:
            on_stage(record)

    tweet_segments = []
    segment_counts = Counter()
    lexical_errors = Counter()

    lexed_rows = []
    example = None

    for row in df.itertuples(index=False):
        lexed = lex_tweet(row.text)
        lexical_errors.update(lexed.errors)
        lexed_rows.append((row, lexed.text))

        if example is None and lexed.errors and lexed.text != str(row.text):
            example = {"before": str(row.text)[:200], "after": lexed.text[:200]}

    stage(
        "Lexer",
        f"{sum(lexical_errors.values()):,} lexical errors recovered "
        f"in {len(df):,} tweets",
        f"{sum(lexical_errors.values()):,}",
        "errors recovered",
        {"errors": dict(lexical_errors.most_common()), "example": example}
    )

    for row, clean_text in lexed_rows:
        segments = segment_tweet(clean_text)
        segment_counts.update(segments)
        tweet_segments.append((row.tweet_id, row.timestamp, segments))

    total_segments = sum(segment_counts.values())

    stage(
        "Segmentation",
        f"{total_segments:,} segments from {len(df):,} tweets "
        f"({total_segments / max(len(df), 1):.1f} per tweet)",
        compact(total_segments),
        "segments",
        {
            "example": (
                tweet_segments[0][2][:8] if tweet_segments else []
            )
        }
    )

    segment_index = build_segment_index(tweet_segments)

    stage(
        "Segment index",
        f"{len(segment_index):,} unique segments indexed",
        compact(len(segment_index)),
        "unique segments",
    )

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

    stage(
        "Burst detection",
        f"{len(bursty):,} bursty segments, top {len(candidates):,} "
        f"kept as candidates",
        f"{len(bursty):,}",
        "bursty segments",
    )

    segments = list(candidates.keys())
    symbol_table = AdaptiveSymbolTable()

    if not historical_mode:
        symbol_table.learn(segments, segment_index, tweet_segments)

    known_segments = symbol_table.filter_segments(segments)

    if known_segments:
        segments = known_segments

    stage(
        "Adaptive symbol table",
        f"{len(segments):,} segments kept "
        f"({len(symbol_table.learned)} emerging phrases learned "
        f"beyond Wikipedia)",
        f"{len(segments):,}",
        "segments kept",
        {"learned": sorted(symbol_table.learned)[:20]}
    )

    clusters = cluster_segments(segments)

    stage(
        "Clustering",
        f"{len(clusters):,} clusters formed",
        f"{len(clusters):,}",
        "clusters"
    )

    events = []

    for cluster in clusters:
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

    stage(
        "Summarization",
        f"{len(events):,} events summarized",
        f"{len(events):,}",
        "events"
    )

    optimizer_info = None

    if not historical_mode:
        events, optimizer_info = optimize_events(
            events, segment_index, tweet_segments
        )
        stage(
            "Event optimizer",
            f"{optimizer_info['before']} events, "
            f"{optimizer_info['after_merge']} after merging duplicates, "
            f"{optimizer_info['after_noise']} after removing noise",
            f"{optimizer_info['after_noise']:,}",
            "events kept",
        )
    else:
        stage(
            "Event optimizer",
            "Skipped (timestamps unknown)",
            "-",
            "skipped"
        )

    novelty = None

    if not historical_mode:
        novelty = analyze_event_novelty(
            df.to_dict("records"),
            {event["summary"]: event["segments"] for event in events},
            now=dataset_end_time(df)
        )

    stage(
        "Novelty",
        "Skipped (timestamps unknown)" if novelty is None
        else f"{len(novelty)} events classified against the last day",
        "-" if novelty is None else f"{len(novelty):,}",
        "skipped" if novelty is None else "events classified",
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

    stage(
        "Evaluation",
        f"{len(events)} events, "
        f"{evaluation['event_coverage'] * 100:.1f}% of tweets covered",
        f"{evaluation['event_coverage'] * 100:.1f}%",
        "tweets covered",
    )

    return {
        "stages": stages,
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
