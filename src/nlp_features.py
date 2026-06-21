"""
NLP feature extraction from Transaction_Description.

Two approaches combined:
  1. Keyword risk scoring — domain knowledge about high-risk transaction descriptions
  2. TF-IDF — learns which words statistically co-occur with fraud in training data

Why both?
  - Keywords are fast, interpretable, and work with zero training data
  - TF-IDF captures patterns keywords miss and generalises to unseen descriptions
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import re


# Words/phrases associated with fraud in banking transaction descriptions
HIGH_RISK_KEYWORDS = {
    'wire transfer': 3, 'international transfer': 3, 'overseas payment': 3,
    'cryptocurrency': 3, 'crypto': 3, 'bitcoin': 3,
    'gift card': 2, 'casino': 2, 'gambling': 2, 'betting': 2,
    'bulk cash': 2, 'rapid transfer': 2, 'unknown merchant': 2,
    'third party': 2, 'card not present': 2,
    'atm withdrawal': 1, 'online transaction': 1, 'contactless': 1,
    'transfer': 1, 'withdrawal': 1,
}

LOW_RISK_KEYWORDS = {
    'salary': -2, 'payroll': -2, 'pension': -2,
    'grocery': -1, 'supermarket': -1, 'utility': -1,
    'electricity': -1, 'insurance': -1, 'school fee': -1,
    'rent': -1, 'medical': -1, 'fuel': -1,
}


def keyword_risk_score(description: str) -> float:
    """Rule-based fraud risk score from transaction description text."""
    if not isinstance(description, str):
        return 0.0
    text = description.lower()
    score = 0.0
    for phrase, weight in HIGH_RISK_KEYWORDS.items():
        if phrase in text:
            score += weight
    for phrase, weight in LOW_RISK_KEYWORDS.items():
        if phrase in text:
            score += weight  # weight is negative for low-risk
    return float(score)


def add_keyword_features(df: pd.DataFrame, desc_col: str = 'Transaction_Description') -> pd.DataFrame:
    """Add keyword-based NLP features to the dataframe."""
    df = df.copy()
    texts = df[desc_col].fillna('').astype(str)

    df['desc_risk_score']    = texts.apply(keyword_risk_score)
    df['desc_has_transfer']  = texts.str.lower().str.contains('transfer').astype(int)
    df['desc_has_crypto']    = texts.str.lower().str.contains('crypto|bitcoin|coin').astype(int)
    df['desc_has_cash']      = texts.str.lower().str.contains('cash|atm|withdrawal').astype(int)
    df['desc_has_online']    = texts.str.lower().str.contains('online|internet|card not present').astype(int)
    df['desc_has_overseas']  = texts.str.lower().str.contains('international|overseas|wire|foreign').astype(int)
    df['desc_word_count']    = texts.apply(lambda x: len(x.split()))
    df['desc_is_generic']    = texts.str.lower().isin(
        ['pos transaction', 'atm withdrawal', 'online transaction',
         'card transaction', 'debit transaction']
    ).astype(int)

    return df


class TFIDFFraudFeatures:
    """
    Fits TF-IDF on training descriptions and transforms to dense feature matrix.
    Limited to top-N components to avoid curse of dimensionality.
    """
    def __init__(self, max_features: int = 50):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),          # unigrams + bigrams
            min_df=2,                    # ignore very rare terms
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'[a-z]{2,}', # only alphabetic tokens
        )
        self.feature_names_: list = []

    def fit(self, descriptions: pd.Series) -> 'TFIDFFraudFeatures':
        texts = descriptions.fillna('').astype(str)
        self.vectorizer.fit(texts)
        self.feature_names_ = [f'tfidf_{t}' for t in self.vectorizer.get_feature_names_out()]
        return self

    def transform(self, descriptions: pd.Series) -> pd.DataFrame:
        texts = descriptions.fillna('').astype(str)
        matrix = self.vectorizer.transform(texts).toarray()
        return pd.DataFrame(matrix, columns=self.feature_names_, index=descriptions.index)

    def fit_transform(self, descriptions: pd.Series) -> pd.DataFrame:
        return self.fit(descriptions).transform(descriptions)
