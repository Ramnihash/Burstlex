import html
import re
import unicodedata
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field

RETWEET_PREFIX = re.compile(r"^\s*rt\s+@\w+:?\s*", re.IGNORECASE)
ELONGATED = re.compile(r"([a-zA-Z])\1{2,}")
CASHTAG = re.compile(r"(?<!\w)\$[A-Za-z][A-Za-z0-9]{1,9}\b")
SPLIT_HASHTAG = re.compile(r"#\s+(?=\w)")
SYMBOLS = re.compile(r"[^\w\s#@.,!?'\"/:;&%$()\-]", re.UNICODE)
REPEATED_PUNCT = re.compile(r"([!?.,])\1{1,}")
SPACES = re.compile(r"\s+")
CURLY_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


PROFANITY_FILE = Path(__file__).with_name("profanity.txt")


def load_profanity(path=PROFANITY_FILE):
    words = [
        line.strip().lower()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    words.sort(key=len, reverse=True)

    return re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(word) for word in words) + r")(?:s|es)?(?!\w)",
        re.IGNORECASE
    )


PROFANITY = load_profanity()


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

    text, count = CASHTAG.subn(" ", text)
    errors["cashtag"] += count

    text, count = PROFANITY.subn(" ", text)
    errors["profanity"] += count

    text, count = SYMBOLS.subn(" ", text)
    errors["symbol_or_emoji"] += count

    text, count = REPEATED_PUNCT.subn(r"\1", text)
    errors["repeated_punctuation"] += count

    text = SPACES.sub(" ", text).strip()

    return LexResult(text=text, errors=+errors)
