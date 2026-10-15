from preprocessing.lexer import lex_tweet


def test_elongated_words_are_collapsed():
    result = lex_tweet("Bidennn is sooo done")
    assert result.text == "Biden is so done"
    assert result.errors["elongated_word"] == 2


def test_html_entity_and_retweet_prefix():
    result = lex_tweet("RT @cnn: tom &amp; jerry")
    assert result.text == "tom & jerry"
    assert result.errors["html_entity"] == 1
    assert result.errors["retweet_prefix"] == 1


def test_split_hashtag_is_repaired():
    assert lex_tweet("# MAGA rally").text == "#MAGA rally"


def test_emoji_removed_and_digits_kept():
    result = lex_tweet("25th amendment 😂😂")
    assert result.text == "25th amendment"
    assert result.errors["symbol_or_emoji"] == 2


def test_curly_quotes_are_straightened():
    assert lex_tweet("Biden’s plan").text == "Biden's plan"


def test_clean_text_has_no_errors():
    result = lex_tweet("plain clean text")
    assert result.text == "plain clean text"
    assert sum(result.errors.values()) == 0


def test_none_input():
    assert lex_tweet(None).text == ""


def test_double_escaped_entity_is_decoded():
    result = lex_tweet("I&amp;#39;ve tried it &amp;amp; it works")
    assert result.text == "I've tried it & it works"
    assert result.errors["html_entity"] == 1


def test_plain_text_has_no_entity_error():
    result = lex_tweet("nothing to decode here")
    assert result.text == "nothing to decode here"
    assert result.errors["html_entity"] == 0
