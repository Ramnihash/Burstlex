import os
import re


DEFAULT_TERMS = {
    "nobel peace prize",
    "justin bieber",
    "presidential debate",
    "national coming out day"
}

TITLES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "wikipedia_titles.txt"
)


def load_titles(path=TITLES_PATH):
    if not os.path.exists(path):
        return set()

    with open(path, encoding="utf-8") as titles_file:
        return {
            line.strip()
            for line in titles_file
            if line.strip()
        }


WIKIPEDIA_TERMS = load_titles() or DEFAULT_TERMS


def normalize_phrase(segment):
    return re.sub(r"\s+", " ", segment.lower()).strip()


def is_wikipedia_phrase(segment, terms=None):
    if terms is None:
        terms = WIKIPEDIA_TERMS

    normalized = normalize_phrase(segment)

    if normalized in terms:
        return True

    if terms is DEFAULT_TERMS:
        return any(term in normalized for term in terms)

    return False


def filter_wikipedia_segments(segments, terms=None):
    return [
        segment
        for segment in segments
        if is_wikipedia_phrase(segment, terms)
    ]


if __name__ == "__main__":
    segments = [
        "nobel peace prize",
        "random phrase",
        "justin bieber",
        "presidential debate",
        "national coming out day celebration"
    ]

    print(filter_wikipedia_segments(segments))
