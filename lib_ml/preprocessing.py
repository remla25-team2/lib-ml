from __future__ import annotations

import re
from typing import Iterable

import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm

import nltk, ssl
try:
    _ = stopwords.words("english")
except LookupError:
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    nltk.download("stopwords", quiet=True)


# _ps = PorterStemmer()
# _stopwords = set(stopwords.words("english")) - {"not"}

# _html_pat = re.compile(r"<[^>]+>")
# _non_letters_pat = re.compile(r"[^a-zA-Z]")


class TextPreprocessor:

    def __init__(self, *, max_features: int = 1420):

        self.ps = PorterStemmer()
        self.stopwords = set(stopwords.words("english")) - {"not"}
        self.html_pat = re.compile(r"<[^>]+>")
        self.non_letters_pat = re.compile(r"[^a-zA-Z]")
        self._vectorizer = CountVectorizer(
            max_features=max_features,
            preprocessor=self.process_item,
            tokenizer=str.split,  # cleaning already lowercases & joins
        )

    # For model training
    def process(self, df):
        corpus = []
        for i in tqdm(range(len(df)), desc="Cleaning text"):
            corpus.append(self.process_item(df['Review'][i]))
        return corpus
    
    def process_item(self, text: str) -> str:
        """Replicates the cleaning pipeline from the restaurant-sentiment repo"""
        text = self.html_pat.sub(" ", text)
        text = self.non_letters_pat.sub(" ", text).lower().split()
        text = [self.ps.stem(w) for w in text if w not in self.stopwords]
        return " ".join(text)

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
        obj = cls()
        obj._vectorizer = joblib.load(path)
        return obj