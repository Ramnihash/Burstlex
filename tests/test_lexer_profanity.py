from preprocessing.lexer import lex_tweet


def test_profanity_is_removed_and_counted():
    result = lex_tweet("This is fucking ridiculous, what the fuck")

    assert "fuck" not in result.text.lower()
    assert result.errors["profanity"] == 2
    assert "ridiculous" in result.text


def test_stretched_and_masked_profanity_is_removed():
    result = lex_tweet("Biden is a f*ck and a shiiit leader")

    assert "f*ck" not in result.text
    assert "shit" not in result.text.lower()
    assert result.errors["profanity"] == 2


def test_clean_words_containing_bad_substrings_are_kept():
    result = lex_tweet("The class assessment passed the Scunthorpe test")

    assert result.text == "The class assessment passed the Scunthorpe test"
    assert "profanity" not in result.errors


def test_profanity_removal_keeps_political_words():
    result = lex_tweet("Trump is a damn liar, shit policy")

    assert "Trump" in result.text
    assert "liar" in result.text
    assert "shit" not in result.text.lower()
