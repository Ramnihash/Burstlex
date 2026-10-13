from data.prepare_wikipedia_titles_api import (
    lookup_titles,
    parse_existing,
    title_variants,
)


def test_title_variants():
    assert title_variants("boston marathon") == {
        "Boston marathon",
        "Boston Marathon",
    }


def test_existing_title_is_found():
    response = {
        "query": {
            "pages": {
                "1": {"title": "Boston Marathon"},
                "-1": {"title": "Zzqx Flurb", "missing": ""},
            }
        }
    }

    assert parse_existing(["Boston Marathon", "Zzqx Flurb"], response) == {
        "Boston Marathon"
    }


def test_redirect_and_normalization_are_followed():
    response = {
        "query": {
            "normalized": [{"from": "boston marathon", "to": "Boston marathon"}],
            "redirects": [{"from": "Boston marathon", "to": "Boston Marathon"}],
            "pages": {"1": {"title": "Boston Marathon"}},
        }
    }

    assert parse_existing(["boston marathon"], response) == {"boston marathon"}


def test_invalid_title_is_ignored():
    response = {"query": {"pages": {"-1": {"title": "x", "invalid": ""}}}}

    assert parse_existing(["x"], response) == set()


class FakeResponse:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self.payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(params["titles"])
        return self.responses.pop(0)


def test_rate_limit_is_retried():
    ok = FakeResponse(200, {"query": {"pages": {"1": {"title": "Boston Marathon"}}}})
    session = FakeSession([FakeResponse(429, headers={"Retry-After": "0"}), ok])
    waits = []
    cache = {}

    found = lookup_titles(
        ["Boston Marathon"], session, cache, sleep=waits.append
    )

    assert found == {"Boston Marathon"}
    assert len(session.calls) == 2


def test_cached_titles_are_not_requested_again():
    session = FakeSession([])
    cache = {"Boston Marathon": True, "Zzqx": False}

    found = lookup_titles(
        ["Boston Marathon", "Zzqx"], session, cache, sleep=lambda seconds: None
    )

    assert found == {"Boston Marathon"}
    assert session.calls == []
