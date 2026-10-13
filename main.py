import pandas as pd
from collections import Counter

from optimization.event_optimizer import optimize_events
from segmentation.segmenter import segment_tweet
from indexing.segment_index import build_segment_index
from burst_detection.bursty_segments import find_bursty_segments
from clustering.event_cluster import cluster_segments
from summarization.event_summary import summarize_event
from segmentation.wikipedia_filter import filter_wikipedia_segments
from evaluation import evaluate_events, count_event_tweets
from novelty.novelty_detector import analyze_event_novelty
from preprocessing.lexer import lex_tweet
from segmentation.adaptive_symbol_table import AdaptiveSymbolTable
from segmentation.wikipedia_filter import WIKIPEDIA_TERMS


def main():
    print("Select Dataset")
    print("1. Sample dataset (24 tweets)")
    print("2. Boston dataset (1000 tweets)")
    print("3. Election 2024 dataset")
    choice = input("Enter choice (1, 2 or 3): ").strip()

    if choice == "3":
        df = pd.read_csv("data/election_tweets.csv")
        min_frequency = 20
        historical_mode = False
    elif choice == "2":
        df = pd.read_csv("data/boston_tweets.csv")
        min_frequency = 10
        historical_mode = True
    else:
        df = pd.read_csv("data/tweets.csv")
        min_frequency = 2
        historical_mode = False

    tweet_segments = []
    segment_counts = Counter()

    lexical_errors = Counter()

    for _, row in df.iterrows():
        lexed = lex_tweet(row["text"])
        lexical_errors.update(lexed.errors)
        segments = segment_tweet(lexed.text)
        segment_counts.update(segments)
        tweet_segments.append(
            (row["tweet_id"], row["timestamp"], segments)
        )

    segment_index = build_segment_index(tweet_segments)

    bursty = find_bursty_segments(
        segment_index,
        min_frequency=min_frequency if choice == "3" else 2
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
            if frequency >= min_frequency
            and not segment.startswith("#")
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
    max_candidates = 2000
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

    wikipedia_segments = symbol_table.filter_segments(segments)
    if wikipedia_segments:
        segments = wikipedia_segments

    clusters = cluster_segments(segments)
    events = []

    for cluster in clusters:
        valid_segments = [
            segment
            for segment in cluster
            if segment in candidates
        ]

        if not valid_segments:
            continue

        summary = summarize_event(valid_segments)
        frequency = count_event_tweets(
            valid_segments,
            segment_index
        )
        score = max(
            candidates[segment]["score"]
            for segment in valid_segments
        )

        events.append({
            "summary": summary,
            "frequency": frequency,
            "score": score,
            "segments": valid_segments
        })

    events.sort(
        key=lambda event: (
            event["frequency"],
            event["score"]
        ),
        reverse=True
    )

    optimizer_info = None

    if not historical_mode:
        events, optimizer_info = optimize_events(
            events,
            segment_index,
            tweet_segments
        )

    print(f"\nDetection Mode: {detection_mode}")
    print("\nAdaptive Symbol Table")
    print("=" * 50)
    print("Wikipedia titles loaded:", len(WIKIPEDIA_TERMS))
    print("Emerging phrases learned:", len(symbol_table.learned))
    print(
        "Examples:",
        ", ".join(sorted(symbol_table.learned)[:15])
    )
    print("\nLexical Errors Recovered")
    print("=" * 50)
    for error_type, count in lexical_errors.most_common():
        print(f"{error_type}: {count}")
    print("Total:", sum(lexical_errors.values()))
    print("\nDetected Events")
    print("=" * 50)

    if optimizer_info:
        print("Events before optimization:", optimizer_info["before"])
        print("After merging duplicates:", optimizer_info["after_merge"])
        print("After removing noise:", optimizer_info["after_noise"])

    for i, event in enumerate(events, 1):
        print(f"\nEvent {i}")
        print("Event:", event["summary"])
        print("Frequency:", event["frequency"])
        print(f"{score_label}:", round(event["score"], 2))
        print(
            "Related Segments:",
            ", ".join(event["segments"][:8])
        )

    print("\nNovelty Analysis")
    print("=" * 50)

    if historical_mode:
        print(
            "Novelty Status: Unavailable for the Boston dataset "
            "because timestamps are unknown"
        )
    else:
        event_segments = {
            event["summary"]: event["segments"]
            for event in events
        }

        dataset_end = pd.to_datetime(df["timestamp"]).max().to_pydatetime()

        novelty_results = analyze_event_novelty(
            df.to_dict("records"),
            event_segments,
            now=dataset_end
        )

        for event_name, details in novelty_results.items():
            print("Event:", event_name)
            print("Previous Window:", details["previous_count"])
            print("Current Window:", details["current_count"])
            print("Status:", details["status"])
            print()

    evaluation_events = [
        {
            "frequency": event["frequency"],
            "burst_score": (
                event["score"] if not historical_mode else 0
            )
        }
        for event in events
    ]

    evaluation = evaluate_events(df, evaluation_events)

    print("\nEvaluation Results")
    print("=" * 50)
    print("Total Tweets:", evaluation["total_tweets"])
    print("Total Events:", evaluation["total_events"])
    print("Average Frequency:", round(evaluation["average_frequency"], 2))
    print(
        "Events per 100 Tweets:",
    round(evaluation["events_per_100_tweets"], 2)
    )
    print(
        "Event Coverage:",
        round(evaluation["event_coverage"] * 100, 2),
    "%"
    )
    print("Highest Event Frequency:", evaluation["highest_frequency"])

    if historical_mode and not bursty:
        average_score = (
            sum(event["score"] for event in events) / len(events)
        if events else 0
        )
        print("Average Frequency Score:", round(average_score, 2))
        print(
            "Temporal Burst Scores: Unavailable because "
            "timestamps are unknown"
            )
    else:
        print(
            "Average Burst Score:",
        round(evaluation["average_burst_score"], 2)
    )
    print("Highest Burst Score:", evaluation["highest_burst_score"])


if __name__ == "__main__":
    main()
