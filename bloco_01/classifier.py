"""
Intent Classifier for Terminal Agent
Categories: SEARCH | CREATE | QUESTION | UNKNOWN
Also predicts complexity rank (1=simple, 5=complex)
"""

import json
import re
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import joblib


# ─────────────────────────────────────────────
# 1. BUILT-IN SEED DATASET
#    (used until you supply your own JSON)
# ─────────────────────────────────────────────

SEED_DATA = [
    # SEARCH — simple (rank 1-2)
    {"text": "find all python files",                  "label": "SEARCH", "rank": 1},
    {"text": "list files in current directory",        "label": "SEARCH", "rank": 1},
    {"text": "search for TODO comments in code",       "label": "SEARCH", "rank": 2},
    {"text": "grep error in logs",                     "label": "SEARCH", "rank": 1},
    {"text": "locate config file",                     "label": "SEARCH", "rank": 1},
    {"text": "find processes using port 8080",         "label": "SEARCH", "rank": 2},
    # SEARCH — complex (rank 3-5)
    {"text": "search all repos for deprecated api calls and summarize", "label": "SEARCH", "rank": 4},
    {"text": "find memory leaks across services",      "label": "SEARCH", "rank": 4},
    {"text": "audit all cron jobs and find conflicts", "label": "SEARCH", "rank": 5},

    # CREATE — simple (rank 1-2)
    {"text": "create a new file called app.py",        "label": "CREATE", "rank": 1},
    {"text": "make a directory named output",          "label": "CREATE", "rank": 1},
    {"text": "write a hello world script",             "label": "CREATE", "rank": 2},
    {"text": "generate a requirements.txt",            "label": "CREATE", "rank": 2},
    {"text": "create a bash script to backup files",   "label": "CREATE", "rank": 3},
    # CREATE — complex (rank 4-5)
    {"text": "build a REST API with authentication",   "label": "CREATE", "rank": 5},
    {"text": "scaffold a full django project",         "label": "CREATE", "rank": 5},
    {"text": "create a docker-compose with redis and postgres", "label": "CREATE", "rank": 4},
    {"text": "write a CI/CD pipeline for github actions", "label": "CREATE", "rank": 4},

    # QUESTION — simple (rank 1-2)
    {"text": "what is my current directory",           "label": "QUESTION", "rank": 1},
    {"text": "how much disk space is left",            "label": "QUESTION", "rank": 1},
    {"text": "what python version is installed",       "label": "QUESTION", "rank": 1},
    {"text": "which user am i logged in as",           "label": "QUESTION", "rank": 1},
    {"text": "what is the ip address of this machine", "label": "QUESTION", "rank": 2},
    # QUESTION — complex (rank 3-5)
    {"text": "why is this service crashing",           "label": "QUESTION", "rank": 3},
    {"text": "explain what this bash script does",     "label": "QUESTION", "rank": 3},
    {"text": "what are the performance bottlenecks in my app", "label": "QUESTION", "rank": 4},
    {"text": "how do i set up kubernetes with autoscaling", "label": "QUESTION", "rank": 5},

    # UNKNOWN
    {"text": "asdf jkl qwerty",                        "label": "UNKNOWN", "rank": 1},
    {"text": "do the thing",                           "label": "UNKNOWN", "rank": 1},
    {"text": "help",                                   "label": "UNKNOWN", "rank": 1},
    {"text": "ok",                                     "label": "UNKNOWN", "rank": 1},
]


# ─────────────────────────────────────────────
# 2. PREPROCESSOR
# ─────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Light normalization: lowercase, strip punctuation, collapse spaces."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ─────────────────────────────────────────────
# 3. CLASSIFIER CLASS
# ─────────────────────────────────────────────

class IntentClassifier:
    """
    Two-head classifier:
      - Head 1: category  → SEARCH | CREATE | QUESTION | UNKNOWN
      - Head 2: rank      → 1 (simple) … 5 (complex)
    """

    CATEGORIES = ["SEARCH", "CREATE", "QUESTION", "UNKNOWN"]

    def __init__(self):
        self.label_encoder = LabelEncoder()

        # Shared TF-IDF vectorizer
        self._vectorizer = TfidfVectorizer(
            preprocessor=preprocess,
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )

        # Category classifier
        self._cat_clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
        )

        # Rank regressor (also LogReg but on integer targets 1-5)
        self._rank_clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.5,
        )

        self._is_trained = False

    # ── training ──────────────────────────────

    def fit(self, data: list[dict], evaluate: bool = True) -> "IntentClassifier":
        """
        Train on a list of {"text": str, "label": str, "rank": int}.
        """
        texts  = [d["text"]  for d in data]
        labels = [d["label"] for d in data]
        ranks  = [d["rank"]  for d in data]

        X = self._vectorizer.fit_transform(texts)
        y_cat  = self.label_encoder.fit_transform(labels)
        y_rank = np.array(ranks)

        if evaluate and len(data) >= 20:
            self._cross_evaluate(X, y_cat, y_rank, labels)

        self._cat_clf.fit(X, y_cat)
        self._rank_clf.fit(X, y_rank)
        self._is_trained = True
        return self

    def _cross_evaluate(self, X, y_cat, y_rank, raw_labels):
        """Quick train/test split report printed to stdout."""
        X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te, rl_tr, rl_te = train_test_split(
            X, y_cat, y_rank, raw_labels, test_size=0.25, random_state=42, stratify=y_cat
        )
        cat_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        cat_clf.fit(X_tr, yc_tr)
        preds = cat_clf.predict(X_te)
        print("\n── Classification Report (25% hold-out) ──")
        print(classification_report(yc_te, preds,
              target_names=self.label_encoder.classes_))

    # ── inference ─────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Returns:
          {
            "text":       original input,
            "label":      "SEARCH" | "CREATE" | "QUESTION" | "UNKNOWN",
            "rank":       int 1-5,
            "confidence": float 0-1,
            "scores":     { category: probability }
          }
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call .fit() first.")

        X = self._vectorizer.transform([text])
        cat_idx   = self._cat_clf.predict(X)[0]
        rank      = int(self._rank_clf.predict(X)[0])
        proba     = self._cat_clf.predict_proba(X)[0]

        scores = {
            cls: round(float(p), 4)
            for cls, p in zip(self.label_encoder.classes_, proba)
        }

        return {
            "text":       text,
            "label":      self.label_encoder.inverse_transform([cat_idx])[0],
            "rank":       rank,
            "confidence": round(float(max(proba)), 4),
            "scores":     scores,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    # ── persistence ───────────────────────────

    def save(self, path: str = "model.joblib"):
        joblib.dump({
            "vectorizer":    self._vectorizer,
            "cat_clf":       self._cat_clf,
            "rank_clf":      self._rank_clf,
            "label_encoder": self.label_encoder,
        }, path)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str = "model.joblib") -> "IntentClassifier":
        obj = joblib.load(path)
        clf = cls()
        clf._vectorizer    = obj["vectorizer"]
        clf._cat_clf       = obj["cat_clf"]
        clf._rank_clf      = obj["rank_clf"]
        clf.label_encoder  = obj["label_encoder"]
        clf._is_trained    = True
        return clf


# ─────────────────────────────────────────────
# 4. DATASET LOADER
# ─────────────────────────────────────────────

def load_dataset(json_path: str | None = None) -> list[dict]:
    """
    Load training data.
    If json_path is provided, merge with seed data.
    Expected JSON format:
      [{"text": "...", "label": "SEARCH", "rank": 2}, ...]
    """
    data = list(SEED_DATA)

    if json_path:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {json_path}")
        with open(path) as f:
            custom = json.load(f)
        data.extend(custom)
        print(f"Loaded {len(custom)} custom samples from {json_path}")

    print(f"Total training samples: {len(data)}")
    return data


# ─────────────────────────────────────────────
# 5. CONVENIENCE FACTORY
# ─────────────────────────────────────────────

def build_classifier(dataset_path: str | None = None) -> IntentClassifier:
    """One-liner: load data → train → return ready classifier."""
    data = load_dataset(dataset_path)
    clf = IntentClassifier()
    clf.fit(data)
    return clf