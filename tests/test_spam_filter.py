from optimization.event_optimizer import drop_spam_events
from preprocessing.lexer import lex_tweet


def make_index(users):
    index = {"seg a": [{"tweet_id": i} for i in range(len(users))]}
    return index, dict(enumerate(users))


def test_cashtags_are_removed_and_counted():
    result = lex_tweet("Trump wins debate $BOME $XRP and costs $5 each")

    assert "BOME" not in result.text
    assert "XRP" not in result.text
    assert "$5" in result.text
    assert result.errors["cashtag"] == 2


def test_event_dominated_by_one_account_is_spam():
    index, users = make_index(["bot"] * 12 + list(range(100, 113)))
    events = [{"summary": "x", "segments": ["seg a"]}]

    kept, dropped = drop_spam_events(events, index, users)

    assert kept == []
    assert dropped == events


def test_event_with_many_accounts_is_kept():
    index, users = make_index(list(range(30)))
    events = [{"summary": "x", "segments": ["seg a"]}]

    kept, dropped = drop_spam_events(events, index, users)

    assert kept == events
    assert dropped == []


def test_small_events_and_missing_users_are_not_judged():
    index, users = make_index(["bot"] * 5)
    events = [{"summary": "x", "segments": ["seg a"]}]

    assert drop_spam_events(events, index, users)[0] == events
    assert drop_spam_events(events, index, None)[0] == events
