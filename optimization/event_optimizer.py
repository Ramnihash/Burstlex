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


def drop_spam_events(events, segment_index, tweet_users,
                     min_tweets=20, max_user_share=0.4):
    if not tweet_users:
        return list(events), []
    kept, dropped = [], []
    for event in events:
        ids = event_tweet_ids(event, segment_index)
        users = Counter(tweet_users[i] for i in ids if i in tweet_users)
        total = sum(users.values())
        if total >= min_tweets and max(users.values()) / total >= max_user_share:
            dropped.append(event)
        else:
            kept.append(event)
    return kept, dropped


def focus_score(event, segment_index, tweet_day):
    ids = event_tweet_ids(event, segment_index)
    counts = Counter(tweet_day[i] for i in ids if i in tweet_day)
    size = sum(counts.values())
    peak_share = max(counts.values()) / size if size else 0.0
    words = len(event["summary"].split())
    return event["frequency"] * peak_share ** 2 * (1 + 0.25 * (words - 1))


def describe_events(events, segment_index, tweet_segments,
                    tweet_texts=None, tweet_users=None, examples=2):
    tweet_day = {tid: day_of(ts) for tid, ts, _ in tweet_segments}
    day_totals = Counter(tweet_day.values())
    total = len(tweet_day)
    rows = []

    for rank, event in enumerate(events, 1):
        ids = event_tweet_ids(event, segment_index)
        counts = Counter(tweet_day[i] for i in ids if i in tweet_day)
        peak_day, peak_count = counts.most_common(1)[0] if counts else ("", 0)
        size = sum(counts.values())

        user_share = ""
        if tweet_users:
            users = Counter(tweet_users[i] for i in ids if i in tweet_users)
            if users:
                user_share = round(max(users.values()) / sum(users.values()), 3)

        samples = ""
        if tweet_texts:
            picked = sorted(ids)[:examples]
            samples = " || ".join(
                str(tweet_texts.get(i, ""))[:140].replace("\n", " ")
                for i in picked
            )

        lift = ""
        if len(day_totals) >= 2 and ids:
            lift = round(day_lift(ids, tweet_day, day_totals, total), 2)

        rows.append({
            "rank": rank,
            "event": event["summary"],
            "frequency": event["frequency"],
            "burst_score": round(event["score"], 2),
            "lift": lift,
            "peak_day": peak_day,
            "peak_share": round(peak_count / size, 3) if size else "",
            "top_user_share": user_share,
            "days": "; ".join(f"{d}:{counts[d]}" for d in sorted(counts)),
            "example_tweets": samples,
            "is_event": "",
            "duplicate_of": "",
        })

    return rows


def optimize_events(events, segment_index, tweet_segments,
                    overlap=0.7, min_lift=1.25, tweet_users=None):
    merged = merge_duplicate_events(events, segment_index, overlap)
    unspammed, spam = drop_spam_events(merged, segment_index, tweet_users)
    kept, dropped = drop_noise_events(
        unspammed, segment_index, tweet_segments, min_lift
    )
    tweet_day = {tid: day_of(ts) for tid, ts, _ in tweet_segments}
    for event in kept:
        event["focus"] = focus_score(event, segment_index, tweet_day)
    kept.sort(key=lambda e: (e["focus"], e["frequency"]), reverse=True)
    return kept, {
        "before": len(events),
        "after_merge": len(merged),
        "after_spam": len(unspammed),
        "after_noise": len(kept),
        "spam": [e["summary"] for e in spam],
        "dropped": [e["summary"] for e in dropped],
    }
