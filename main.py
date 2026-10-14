import pandas as pd

from pipeline import run_pipeline


def main():
    print("Select Dataset")
    print("1. Sample dataset (1000 tweets)")
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
        min_frequency = 20
        historical_mode = False

    result = run_pipeline(df, min_frequency, historical_mode)
    events = result["events"]
    evaluation = result["evaluation"]
    symbol_table = result["symbol_table"]
    optimizer_info = result["optimizer_info"]
    score_label = result["score_label"]

    print(f"\nDetection Mode: {result['detection_mode']}")

    print("\nAdaptive Symbol Table")
    print("=" * 50)
    print("Emerging phrases learned:", len(symbol_table.learned))
    print("Examples:", ", ".join(sorted(symbol_table.learned)[:15]))

    print("\nLexical Errors Recovered")
    print("=" * 50)

    for error_type, count in result["lexical_errors"].most_common():
        print(f"{error_type}: {count}")

    print("Total:", sum(result["lexical_errors"].values()))

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
        print("Related Segments:", ", ".join(event["segments"][:8]))

    print("\nNovelty Analysis")
    print("=" * 50)

    if result["novelty"] is None:
        print(
            "Novelty Status: Unavailable for the Boston dataset "
            "because timestamps are unknown"
        )
    else:
        for event_name, details in result["novelty"].items():
            print("Event:", event_name)
            print("Previous Window:", details["previous_count"])
            print("Current Window:", details["current_count"])
            print("Status:", details["status"])
            print()

    print("\nEvaluation Results")
    print("=" * 50)
    print("Total Tweets:", evaluation["total_tweets"])
    print("Total Events:", evaluation["total_events"])
    print("Average Frequency:", round(evaluation["average_frequency"], 2))
    print(
        "Events per 100 Tweets:",
        round(evaluation["events_per_100_tweets"], 2)
    )
    print("Event Coverage:", round(evaluation["event_coverage"] * 100, 2), "%")
    print("Highest Event Frequency:", evaluation["highest_frequency"])

    if result["historical_mode"]:
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
        print("Average Burst Score:", round(evaluation["average_burst_score"], 2))
        print("Highest Burst Score:", evaluation["highest_burst_score"])


if __name__ == "__main__":
    main()
