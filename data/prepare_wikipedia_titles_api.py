import json
import os
import sys
import time
from collections import Counter

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation.segmenter import segment_tweet

API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "BurstlexStudentProject/1.0 (university course project; contact narotham17@gmail.com)"}
BATCH_SIZE = 50


def count_segments(dataset_path):
    df = pd.read_csv(dataset_path)
    counts = Counter()

    for text in df["text"]:
        counts.update(
            segment
            for segment in segment_tweet(text)
            if not segment.startswith("#")
        )

    return counts


def title_variants(segment):
    first_upper = segment[:1].upper() + segment[1:]
    return {first_upper, segment.title()}


def parse_existing(requested, response):
    query = response.get("query", {})
    normalized = {item["from"]: item["to"] for item in query.get("normalized", [])}
    redirects = {item["from"]: item["to"] for item in query.get("redirects", [])}
    existing_pages = {
        page["title"]
        for page in query.get("pages", {}).values()
        if "missing" not in page and "invalid" not in page
    }

    found = set()

    for title in requested:
        resolved = normalized.get(title, title)
        resolved = redirects.get(resolved, resolved)

        if resolved in existing_pages:
            found.add(title)

    return found


def load_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}

    with open(cache_path, encoding="utf-8") as cache_file:
        return json.load(cache_file)


def save_cache(cache_path, cache):
    with open(cache_path, "w", encoding="utf-8") as cache_file:
        json.dump(cache, cache_file)


def request_batch(batch, session, sleep=time.sleep):
    for attempt in range(6):
        response = session.get(
            API_URL,
            params={
                "action": "query",
                "titles": "|".join(batch),
                "redirects": 1,
                "format": "json",
            },
            headers=HEADERS,
            timeout=30,
        )

        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 15 * (attempt + 1)))
            print("Rate limited, waiting", wait, "seconds")
            sleep(wait)
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError("Wikipedia kept refusing requests. Run the script again later.")


def lookup_titles(titles, session, cache, cache_path=None, sleep=time.sleep):
    pending = sorted(title for title in titles if title not in cache)

    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]
        found = parse_existing(batch, request_batch(batch, session, sleep))

        for title in batch:
            cache[title] = title in found

        if cache_path:
            save_cache(cache_path, cache)

        print("Checked", min(start + BATCH_SIZE, len(pending)), "of", len(pending))
        sleep(1.0)

    return {title for title in titles if cache.get(title)}


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: py data\\prepare_wikipedia_titles_api.py "
            "<dataset_csv> <output_txt> [min_frequency]"
        )
        return

    dataset_path, output_path = sys.argv[1:3]
    min_frequency = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    counts = count_segments(dataset_path)
    candidates = [
        segment
        for segment, frequency in counts.items()
        if frequency >= min_frequency
    ]

    variant_to_segment = {}

    for segment in candidates:
        for variant in title_variants(segment):
            variant_to_segment[variant] = segment

    print("Candidate segments:", len(candidates))
    print("Titles to check:", len(variant_to_segment))

    cache_path = output_path + ".cache.json"
    cache = load_cache(cache_path)

    with requests.Session() as session:
        found = lookup_titles(
            variant_to_segment.keys(), session, cache, cache_path
        )

    matched = {variant_to_segment[title] for title in found}

    with open(output_path, "w", encoding="utf-8") as output:
        for title in sorted(matched):
            output.write(title + "\n")

    print("Matched Wikipedia titles:", len(matched))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
