"""Text normalisation shared by training and the app.

SMS spam leans heavily on URLs, phone numbers, prices and shouting, so rather
than stripping those out we replace them with placeholder tokens the model can
learn from.
"""

import re

_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_MONEY = re.compile(r"[£$€]\s?\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s?(?:p|pence|gbp|usd)\b", re.IGNORECASE)
_PHONE = re.compile(r"\b\d{5,}\b")
_NUMBER = re.compile(r"\b\d+\b")
_NON_WORD = re.compile(r"[^a-z_ ]+")
_SPACES = re.compile(r"\s+")


def normalise(message: str) -> str:
    """Lowercase a message and swap noisy spans for placeholder tokens."""
    text = str(message)
    text = _URL.sub(" _url_ ", text)
    text = _EMAIL.sub(" _email_ ", text)
    text = _MONEY.sub(" _money_ ", text)
    text = _PHONE.sub(" _phone_ ", text)
    text = _NUMBER.sub(" _num_ ", text)
    text = text.lower()
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
