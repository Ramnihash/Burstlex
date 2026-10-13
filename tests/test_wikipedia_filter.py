from segmentation.wikipedia_filter import (
    DEFAULT_TERMS,
    filter_wikipedia_segments,
    is_wikipedia_phrase,
    load_titles,
)


def test_default_terms_allow_partial_match():
    assert is_wikipedia_phrase(
        "national coming out day celebration",
        DEFAULT_TERMS
    )


def test_full_titles_require_exact_match():
    titles = {"boston marathon", "chris christie"}

    assert is_wikipedia_phrase("boston marathon", titles)
    assert not is_wikipedia_phrase("boston marathon bombing", titles)


def test_phrase_is_normalized():
    titles = {"boston marathon"}

    assert is_wikipedia_phrase("  Boston   Marathon ", titles)


def test_filter_keeps_only_titles():
    titles = {"boston marathon", "chris christie"}
    segments = ["boston marathon", "random phrase", "chris christie"]

    assert filter_wikipedia_segments(segments, titles) == [
        "boston marathon",
        "chris christie",
    ]


def test_missing_titles_file_returns_empty_set(tmp_path):
    assert load_titles(str(tmp_path / "missing.txt")) == set()


def test_titles_file_is_loaded(tmp_path):
    path = tmp_path / "titles.txt"
    path.write_text("boston marathon\n\nchris christie\n", encoding="utf-8")

    assert load_titles(str(path)) == {"boston marathon", "chris christie"}
