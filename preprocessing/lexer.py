import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

RETWEET_PREFIX = re.compile(r"^\s*rt\s+@\w+:?\s*", re.IGNORECASE)
ELONGATED = re.compile(r"([a-zA-Z])\1{2,}")
SPLIT_HASHTAG = re.compile(r"#\s+(?=\w)")
SYMBOLS = re.compile(r"[^\w\s#@.,!?'\"/:;&%$()\-]", re.UNICODE)
REPEATED_PUNCT = re.compile(r"([!?.,])\1{1,}")
SPACES = re.compile(r"\s+")
CURLY_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


@dataclass
class LexResult:
    text: str
    errors: Counter = field(default_factory=Counter)


def lex_tweet(text):
    errors = Counter()
    text = "" if text is None else str(text)

    normalized = unicodedata.normalize("NFKC", text)
    if normalized != text:
        errors["unicode_normalized"] += 1
    text = normalized

    straightened = text.translate(CURLY_QUOTES)
    if straightened != text:
        errors["curly_quote"] += 1
    text = straightened

    unescaped = html.unescape(text)
    if unescaped != text:
        errors["html_entity"] += 1
    text = unescaped

    text, count = RETWEET_PREFIX.subn("", text)
    errors["retweet_prefix"] += count

    text, count = SPLIT_HASHTAG.subn("#", text)
    errors["split_hashtag"] += count

    text, count = ELONGATED.subn(r"\1", text)
    errors["elongated_word"] += count

    text, count = SYMBOLS.subn(" ", text)
    errors["symbol_or_emoji"] += count

    text, count = REPEATED_PUNCT.subn(r"\1", text)
    errors["repeated_punctuation"] += count

    text = SPACES.sub(" ", text).strip()

    return LexResult(text=text, errors=+errors)
