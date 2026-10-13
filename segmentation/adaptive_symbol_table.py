import re
from collections import Counter

from optimization.event_optimizer import day_lift, day_of
from segmentation.wikipedia_filter import is_wikipedia_phrase

ASCII_PHRASE = re.compile(r"^[a-z0-9]+( [a-z0-9]+)+$")


class AdaptiveSymbolTable:
    def __init__(self, wikipedia_terms=None):
        self.wikipedia_terms = wikipedia_terms
        self.learned = set()

    def is_wikipedia(self, segment):
        return is_wikipedia_phrase(segment, self.wikipedia_terms)

    def contains(self, segment):
        return self.is_wikipedia(segment) or segment in self.learned

    def learn(self, candidates, segment_index, tweet_segments,
              min_frequency=25, min_lift=2.5):
        tweet_day = {tid: day_of(ts) for tid, ts, _ in tweet_segments}
        day_totals = Counter(tweet_day.values())

        if len(day_totals) < 2:
            return set()

        total = len(tweet_day)
        learned = set()

        for segment in candidates:
            if self.is_wikipedia(segment) or not ASCII_PHRASE.match(segment):
                continue

            ids = {p["tweet_id"] for p in segment_index.get(segment, [])}

            if len(ids) < min_frequency:
                continue

            if day_lift(ids, tweet_day, day_totals, total) >= min_lift:
                learned.add(segment)

        self.learned |= learned
        return learned

    def filter_segments(self, segments):
        return [segment for segment in segments if self.contains(segment)]
