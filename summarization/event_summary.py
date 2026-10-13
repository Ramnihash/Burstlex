
import re

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


NOISE_TERMS = {
    "rt", "amp", "today", "breaking", "news", "latest",
    "update", "people", "time", "via", "http", "https",
    "com", "running", "go", "goes", "get", "got", "one",
    "two", "year", "old"
}


def clean_phrase(phrase):
    words = re.findall(r"[a-z0-9]+", phrase.lower())
    words = [word for word in words if word not in NOISE_TERMS]
    return " ".join(words)


def summarize_event(cluster):
    if not cluster:
        return ""

    unique_phrases = list(dict.fromkeys(
        phrase.strip().lower()
        for phrase in cluster
        if phrase and phrase.strip()
    ))

    if not unique_phrases:
        return ""

    cleaned_phrases = [
        clean_phrase(phrase)
        for phrase in unique_phrases
    ]

    valid_indices = [
        index
        for index, phrase in enumerate(cleaned_phrases)
        if phrase
    ]

    if not valid_indices:
        return unique_phrases[0]

    if len(valid_indices) == 1:
        return cleaned_phrases[valid_indices[0]]

    phrases = [cleaned_phrases[index] for index in valid_indices]

    matrix = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    ).fit_transform(phrases)

    similarity = cosine_similarity(matrix)
    graph = nx.from_numpy_array(similarity)
    scores = nx.pagerank(graph)

    ranked_indices = sorted(
        range(len(phrases)),
        key=lambda index: (
            scores[index],
            len(set(phrases[index].split())),
            -len(phrases[index])
        ),
        reverse=True
    )

    best_phrase = phrases[ranked_indices[0]]
    best_words = best_phrase.split()

    if len(best_words) > 5:
        best_phrase = " ".join(best_words[:5])

    return best_phrase


if __name__ == "__main__":
    cluster = [
        "nobel peace prize",
        "nobel peace prize announced today",
        "peace prize ceremony today"
    ]

    print("Event Summary:")
    print(summarize_event(cluster))
