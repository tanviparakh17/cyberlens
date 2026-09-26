import os                                    # file paths handle karne ke liye
import joblib                                # trained model ko save/load karne ke liye
import pandas as pd                          # data ko table (DataFrame) format mein rakhne ke liye
from sklearn.ensemble import RandomForestClassifier   # ML algorithm jo hum use kar rahe hain
from sklearn.model_selection import train_test_split  # data ko train/test mein baantne ke liye
from sklearn.metrics import accuracy_score, classification_report  # model ki performance check karne ke liye

from utils.feature_extraction import extract_url_features  # humara feature-extraction function import kiya

# Model file kahan save/load hogi -- agents folder ke ek level upar, models/ folder mein
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "url_model.joblib")

# Real dataset kahan se load hogi (prepare_data.py se bani final file)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "phishing_urls.csv")


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
    pred_label = clf.predict(X)[0]
    confidence = float(proba[pred_label])

    return {
        "url": url,
        "verdict": "phishing" if pred_label == 1 else "safe",
        "confidence": round(confidence, 3),
        "features": features,
        "model": clf,
        "feature_row": X,
    }


if __name__ == "__main__":
    train()
    test_url = "http://paypal-account-verify.suspend-secure-login.xyz"
    result = predict(test_url)
    print(result)