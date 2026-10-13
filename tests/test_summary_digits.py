from summarization.event_summary import summarize_event


def test_digits_are_kept_in_summary():
    assert summarize_event(["25th amendment"]) == "25th amendment"
    assert summarize_event(["january 6th"]) == "january 6th"
