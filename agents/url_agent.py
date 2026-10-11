import os
from functools import lru_cache
from urllib.parse import urlparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from utils.feature_extraction import OFFICIAL_DOMAINS, extract_url_features, get_registered_domain

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")

# Where the trained model is saved/loaded (models/ folder, one level above agents/)
MODEL_PATH = os.path.join(BASE_DIR, "models", "url_model.joblib")

# Training data created by prepare_data.py (columns "url" and "label": 0 = safe, 1 = phishing)
DATA_PATH = os.path.join(BASE_DIR, "data", "phishing_urls.csv")

# Allowlists: the generated top-10k list, plus an optional hand-written list
TRUSTED_PATH = os.path.join(BASE_DIR, "data", "trusted_domains.txt")
KNOWN_GOOD_PATH = os.path.join(BASE_DIR, "data", "known_good_domains.txt")

# When the model alone says "phishing" with less confidence than this,
# the verdict is "suspicious" (the model is unsure) instead of "phishing".
SUSPICIOUS_BELOW = 0.70


def _read_domain_file(path: str) -> set:
    """One registered domain per line; blank lines and '#' comments are ignored."""
    if not os.path.exists(path):
        return set()
    domains = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip().lower()
            if line:
                domains.add(line)
    return domains


@lru_cache(maxsize=1)
def _load_trusted() -> frozenset:
    return frozenset(
        _read_domain_file(TRUSTED_PATH)
        | _read_domain_file(KNOWN_GOOD_PATH)
        | set(OFFICIAL_DOMAINS)
    )


def _rule_override(features: dict):
    """Strong signals override the ML model. Returns (verdict, reason) or None."""
    if features["has_ip_address"] and features["has_suspicious_keyword"]:
        return "phishing", "IP address host + suspicious keyword"
    if features["brand_impersonation"] or features["homoglyph_brand"]:
        return "phishing", "brand name impersonation in domain/subdomain"
    if features["suspicious_tld"] and features["has_suspicious_keyword"]:
        return "phishing", "suspicious TLD + suspicious keyword"
    return None


def _decide(proba, features: dict, is_trusted: bool):
    """
    Combines the ML probabilities, the rules and the allowlist.
    proba = [P(safe), P(phishing)]. Returns (verdict, confidence, source, reason).
    """
    pred_label = 1 if proba[1] >= proba[0] else 0
    verdict = "phishing" if pred_label == 1 else "safe"
    confidence = float(proba[pred_label])
    source, reason = "ml", None

    override = _rule_override(features)
    if override:
        verdict, reason = override
        confidence = max(float(proba[1]), 0.85)
        source = "rule"
    elif is_trusted and not features["has_ip_address"]:
        verdict, confidence = "safe", max(float(proba[0]), 0.95)
        source, reason = "allowlist", "domain in trusted list"
    elif verdict == "phishing" and confidence < SUSPICIOUS_BELOW:
        verdict = "suspicious"
        reason = f"model is unsure ({confidence:.0%} phishing)"

    return verdict, confidence, source, reason


def load_training_data() -> pd.DataFrame:
    """Loads data/phishing_urls.csv, created by prepare_data.py."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            f"Run 'python prepare_data.py' first to generate it."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} URLs "
          f"({(df['label'] == 1).sum()} phishing, {(df['label'] == 0).sum()} safe)")

    return df[["url", "label"]]


def get_domain(url):
    try:
        return urlparse(url if "://" in url else "http://" + url).hostname
    except Exception:
        return url


def train():
    df = load_training_data()
    df["domain"] = df["url"].apply(get_domain)

    # Domain-based split: a domain appears in train OR test, never both
    unique_domains = list(df["domain"].unique())
    train_domains, test_domains = train_test_split(
        unique_domains, test_size=0.3, random_state=42
    )

    train_df = df[df["domain"].isin(train_domains)]
    test_df = df[df["domain"].isin(test_domains)]

    print(f"Train: {len(train_df)} URLs from {len(train_domains)} domains")
    print(f"Test: {len(test_df)} URLs from {len(test_domains)} domains")

    X_train = pd.DataFrame([extract_url_features(u) for u in train_df["url"]])
    y_train = train_df["label"].reset_index(drop=True)

    X_test = pd.DataFrame([extract_url_features(u) for u in test_df["url"]])
    y_test = test_df["label"].reset_index(drop=True)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("Test set Accuracy (domain-separated):", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": clf, "feature_names": list(X_train.columns)}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return clf


def _load_model():
    if not os.path.exists(MODEL_PATH):
        train()
    return joblib.load(MODEL_PATH)


def predict(url: str) -> dict:
    bundle = _load_model()
    clf = bundle["model"]
    feature_names = bundle["feature_names"]

    features = extract_url_features(url)
    X = pd.DataFrame([features])[feature_names]

    proba = clf.predict_proba(X)[0]
    is_trusted = get_registered_domain(url) in _load_trusted()
    verdict, confidence, source, reason = _decide(proba, features, is_trusted)

    return {
        "url": url,
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "source": source,
        "reason": reason,
        "features": features,
        "model": clf,
        "feature_row": X,
    }


if __name__ == "__main__":
    train()
    test_url = "http://paypal-account-verify.suspend-secure-login.xyz"
    result = predict(test_url)
    print(result)