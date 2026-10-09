import os                                    # file paths handle karne ke liye
import joblib                                # trained model ko save/load karne ke liye
import pandas as pd                          # data ko table (DataFrame) format mein rakhne ke liye
from sklearn.ensemble import RandomForestClassifier   # ML algorithm jo hum use kar rahe hain
from sklearn.model_selection import train_test_split  # data ko train/test mein baantne ke liye
from sklearn.metrics import accuracy_score, classification_report  # model ki performance check karne ke liye

 # humara feature-extraction function import kiya
from utils.feature_extraction import extract_url_features, get_registered_domain
# Model file kahan save/load hogi -- agents folder ke ek level upar, models/ folder mein
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "url_model.joblib")

# Real dataset kahan se load hogi (prepare_data.py se bani final file)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "phishing_urls.csv")

from functools import lru_cache

TRUSTED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trusted_domains.txt")


@lru_cache(maxsize=1)
def _load_trusted() -> frozenset:
    if not os.path.exists(TRUSTED_PATH):
        return frozenset()
    with open(TRUSTED_PATH) as f:
        return frozenset(line.strip() for line in f if line.strip())


def _rule_override(features: dict):
    """Strong signals pe ML ko override karta hai. (verdict, reason) ya None."""
    if features["has_ip_address"] and features["has_suspicious_keyword"]:
        return "phishing", "IP address host + suspicious keyword"
    if features["brand_impersonation"] or features["homoglyph_brand"]:
        return "phishing", "brand name impersonation in domain/subdomain"
    if features["suspicious_tld"] and features["has_suspicious_keyword"]:
        return "phishing", "suspicious TLD + suspicious keyword"
    return None

def load_training_data() -> pd.DataFrame:
    """
    prepare_data.py se generate hui final dataset (data/phishing_urls.csv) load karta hai.
    Ye file mein already columns "url" aur "label" hain (0=safe, 1=phishing).
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            f"Run 'python prepare_data.py' first to generate it."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} URLs "
          f"({(df['label'] == 1).sum()} phishing, {(df['label'] == 0).sum()} safe)")

    return df[["url", "label"]]


from sklearn.model_selection import train_test_split, cross_val_score
from urllib.parse import urlparse

def get_domain(url):
    try:
        return urlparse(url if "://" in url else "http://" + url).hostname
    except:
        return url

def train():
    df = load_training_data()
    df["domain"] = df["url"].apply(get_domain)

    # ---- Domain-based split: ek domain sirf train YA test mein aayega, dono mein nahi ----
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
    pred_label = int(clf.predict(X)[0])
    verdict = "phishing" if pred_label == 1 else "safe"
    confidence = float(proba[pred_label])
    source, reason = "ml", None

    override = _rule_override(features)
    if override:
        verdict, reason = override
        confidence = max(float(proba[1]), 0.85)
        source = "rule"
    elif get_registered_domain(url) in _load_trusted() and not features["has_ip_address"]:
        verdict, confidence = "safe", max(float(proba[0]), 0.95)
        source, reason = "allowlist", "domain in trusted top-10k list"

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