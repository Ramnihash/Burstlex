from collections import Counter

from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()


def stem_key(segment):
    return frozenset(_stemmer.stem(word) for word in segment.split())


def event_tweet_ids(event, segment_index):
    ids = set()
    for segment in event["segments"]:
        for posting in segment_index.get(segment, []):
            ids.add(posting["tweet_id"])
    return ids


def day_of(timestamp):
    return str(timestamp)[:10]


def day_lift(tweet_ids, tweet_day, day_totals, total_tweets):
    if not tweet_ids:
        return 0.0
    counts = Counter(tweet_day[i] for i in tweet_ids if i in tweet_day)
    size = len(tweet_ids)
    return max(
        (counts[day] / size) / (day_totals[day] / total_tweets)
        for day in counts
    )


def jaccard(first, second):
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def merge_duplicate_events(events, segment_index, overlap=0.7):
    ids = [event_tweet_ids(event, segment_index) for event in events]
    keys = [{stem_key(s) for s in event["segments"]} for event in events]
    parent = list(range(len(events)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for index, event_keys in enumerate(keys):
        for key in event_keys:
            if key in owner:
                parent[find(index)] = find(owner[key])
            else:
                owner[key] = index

    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            if find(i) != find(j) and jaccard(ids[i], ids[j]) >= overlap:
                parent[find(i)] = find(j)

    groups = {}
    for index in range(len(events)):
        groups.setdefault(find(index), []).append(index)

    merged = []
    for members in groups.values():
        members.sort(key=lambda m: events[m]["frequency"], reverse=True)
        lead = events[members[0]]
        segments = []
        for m in members:
            for segment in events[m]["segments"]:
                if segment not in segments:
                    segments.append(segment)
        all_ids = set().union(*(ids[m] for m in members))
        merged.append({
            "summary": lead["summary"],
            "frequency": len(all_ids) if len(members) > 1 else lead["frequency"],
            "score": max(events[m]["score"] for m in members),
            "segments": segments,
        })
    return merged


def drop_noise_events(events, segment_index, tweet_segments, min_lift=1.25):
    tweet_day = {tid: day_of(ts) for tid, ts, _ in tweet_segments}
    day_totals = Counter(tweet_day.values())
    if len(day_totals) < 2:
        return list(events), []
    total = len(tweet_day)
    kept, dropped = [], []
    for event in events:
        lift = day_lift(
            event_tweet_ids(event, segment_index),
            tweet_day, day_totals, total
        )
        (kept if lift >= min_lift else dropped).append(event)
    return kept, dropped


def optimize_events(events, segment_index, tweet_segments,
                    overlap=0.7, min_lift=1.25):
    merged = merge_duplicate_events(events, segment_index, overlap)
    kept, dropped = drop_noise_events(
        merged, segment_index, tweet_segments, min_lift
    )
    kept.sort(key=lambda e: (e["frequency"], e["score"]), reverse=True)
    return kept, {
        "before": len(events),
        "after_merge": len(merged),
        "after_noise": len(kept),
        "dropped": [e["summary"] for e in dropped],
    }
