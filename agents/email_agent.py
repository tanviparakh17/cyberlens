import os
import re
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# File path configurations relative to this module
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "email_model.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "email_scams.csv")

# High-risk security and urgency keywords common in scam emails
URGENCY_KEYWORDS = [
    "urgent", "immediately", "verify", "suspended", "account", "action required",
    "security", "login", "password", "bank", "update", "expire", "alert", "click"
]

def extract_email_features(text: str, vectorizer: TfidfVectorizer = None) -> pd.DataFrame:
    """
    Extracts text-based heuristic indicators and combines them with TF-IDF features.
    Returns a single-row pandas DataFrame compatible with SHAP and Scikit-learn.
    """
    text_clean = str(text) if pd.notna(text) else ""
    text_lower = text_clean.lower()
    
    # 1. Structural and heuristic feature extraction
    num_links = len(re.findall(r'https?://\S+|www\.\S+', text_clean))
    urgency_count = sum(text_lower.count(word) for word in URGENCY_KEYWORDS)
    num_dollar_signs = text_clean.count('$')
    num_exclamations = text_clean.count('!')
    text_length = len(text_clean)
    
    heuristics = {
        "num_links": num_links,
        "urgency_keywords": urgency_count,
        "num_dollar_signs": num_dollar_signs,
        "num_exclamations": num_exclamations,
        "text_length": text_length,
    }
    
    heuristics_df = pd.DataFrame([heuristics])
    
    # 2. Append TF-IDF features if a fitted vectorizer is provided
    if vectorizer is not None:
        tfidf_sparse = vectorizer.transform([text_clean])
        tfidf_df = pd.DataFrame(
            tfidf_sparse.toarray(), 
            columns=[f"tfidf_{col}" for col in vectorizer.get_feature_names_out()]
        )
        features_df = pd.concat([heuristics_df, tfidf_df], axis=1)
    else:
        features_df = heuristics_df
        
    return features_df

def load_training_data() -> pd.DataFrame:
    """
    Loads and validates data/email_scams.csv.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Please create data/email_scams.csv first."
        )
    df = pd.read_csv(DATA_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV dataset must contain 'text' and 'label' columns.")
    return df

def train():
    """
    Trains TF-IDF vectorizer + RandomForestClassifier, validates performance,
    and exports model bundle for inference and XAI.
    """
    df = load_training_data()
    print(f"✅ Loaded {len(df)} emails ({(df['label'] == 1).sum()} phishing, {(df['label'] == 0).sum()} safe)")
    
    # Stratified split to preserve class ratios
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    
    # Fit TF-IDF on top vocabulary terms to maintain SHAP performance and readability
    vectorizer = TfidfVectorizer(max_features=150, stop_words=list(ENGLISH_STOP_WORDS | {"date", "id", "aug", "company", "2026", "2008", "sat", "mon", "tue", "wed", "thu", "fri", "sun"}))
    vectorizer.fit(train_df["text"].fillna(""))
    
    # Build train feature matrix
    X_train_list = [extract_email_features(t, vectorizer) for t in train_df["text"].fillna("")]
    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = train_df["label"].reset_index(drop=True)
    
    # Build test feature matrix
    X_test_list = [extract_email_features(t, vectorizer) for t in test_df["text"].fillna("")]
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_test = test_df["label"].reset_index(drop=True)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=3, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    print("\nTest Set Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        "model": clf,
        "vectorizer": vectorizer,
        "feature_names": list(X_train.columns)
    }, MODEL_PATH)
    
    print(f"✅ Model saved successfully to {MODEL_PATH}")
    return clf

def _load_model():
    if not os.path.exists(MODEL_PATH):
        train()
    return joblib.load(MODEL_PATH)

SENSITIVE_ASK = re.compile(
    r"(enter|share|send|reply with|provide|confirm).{0,25}\b(otp|pin|cvv|password|aadhaar|pan)\b"
    r"|password (will )?expires?|card number|bank details|account number"
    r"|(registration|processing|clearance) fee|\bpay\b.{0,25}\b(fee|rs|inr)\b|gift cards?"
    r"|verify your|confirm your identity|kyc"
    r"|(account|banking).{0,30}(blocked|suspended|locked|deactivated)",
    re.I,
)
SAFE_PHRASES = re.compile(r"never ask|never share|do not share|don't share|will not ask|won't ask", re.I)

def _apply_rules(text: str, p: float):
    """Rule layer on top of the model: adjusts the phishing probability and explains why."""
    rules = []
    kept = [sn for sn in re.split(r"(?<=[.!?\n])\s+", text) if not SAFE_PHRASES.search(sn)]
    cleaned = " ".join(kept)
    has_link = bool(re.search(r"https?://|www\.", text))
    if SENSITIVE_ASK.search(cleaned):
        p = min(1.0, p + 0.25)
        rules.append("asks for sensitive info / payment / urgent account action (+0.25)")
    elif not has_link:
        p = p * 0.7
        rules.append("no links and no sensitive request (x0.7)")
    return p, rules

def predict(email_text: str) -> dict:
    """
    Predicts whether an email text is phishing or safe.
    Returns interface dict compatible with xai_agent.py and report_agent.py.
    """
    bundle = _load_model()
    clf = bundle["model"]
    vectorizer = bundle["vectorizer"]
    feature_names = bundle["feature_names"]
    
    X = extract_email_features(email_text, vectorizer)[feature_names]
    
    proba = clf.predict_proba(X)[0]
    pred_label = clf.predict(X)[0]
    confidence = float(proba[pred_label])
    
    p_phish, rules = _apply_rules(email_text, float(proba[1]))
    verdict = "phishing" if p_phish >= 0.65 else ("suspicious" if p_phish >= 0.50 else "safe")
    confidence = p_phish if verdict != "safe" else 1 - p_phish
    features_dict = X.to_dict(orient="records")[0]
    
    return {
        "email_text": email_text,
        "verdict": verdict,
        "rules": rules,
        "confidence": round(confidence, 3),
        "features": features_dict,
        "model": clf,
        "feature_row": X,
    }

if __name__ == "__main__":
    train()
    
    # Test sample prediction
    test_email = "URGENT ACTION REQUIRED: Your account access is suspended. Verify immediately at http://update-secure-login.com"
    result = predict(test_email)
    print("\n--- Test Prediction Output ---")
    print(f"Email: {result['email_text']}")
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']}")