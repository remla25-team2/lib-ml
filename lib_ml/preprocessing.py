from __future__ import annotations

import re
from typing import Iterable

import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

_ps = PorterStemmer()
_stopwords = set(stopwords.words("english")) - {"not"}  # keep negation word

_html_pat = re.compile(r"<[^>]+>")
_non_letters_pat = re.compile(r"[^a-zA-Z]")


def _clean(text: str) -> str:
    """Replicates the cleaning pipeline from the restaurant-sentiment repo"""
    text = _html_pat.sub(" ", text)
    text = _non_letters_pat.sub(" ", text).lower().split()
    text = [_ps.stem(w) for w in text if w not in _stopwords]
    return " ".join(text)


class TextPreprocessor:

    def __init__(self, *, max_features: int = 1420):
        self._vectorizer = CountVectorizer(
            max_features=max_features,
            preprocessor=_clean,
            tokenizer=str.split,  # cleaning already lowercases & joins
        )

    # Core API
    def fit(self, texts: Iterable[str]) -> "TextPreprocessor":
        self._vectorizer.fit(texts)
        return self

    def transform(self, texts: Iterable[str]):
        return self._vectorizer.transform(texts)

    def save(self, path: str) -> None:
        joblib.dump(self._vectorizer, path)

    @classmethod
    def load(cls, path: str) -> "TextPreprocessor":
        obj = cls.__new__(cls)
        obj._vectorizer = joblib.load(path)
        return obj