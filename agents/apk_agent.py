"""
CyberLens APK Malware Detection Agent

Analyzes Android APK files with Androguard, then scores the requested
permissions with a RandomForest model and explains the score with SHAP.

Training data:
  1) data/apk_permissions.csv, if it exists (real dataset, see load_training_data)
  2) otherwise synthetic permission profiles (assumed, NOT real malware samples)

Usage:
    python -m agents.apk_agent "path\\to\\file.apk"
    python -m agents.apk_agent --train        (retrain the model only)
"""
import sys
import hashlib
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from androguard.misc import AnalyzeAPK
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "apk_model.joblib")

# Optional real dataset: one row per APK, one 0/1 column per permission
# (e.g. "READ_SMS" or "android.permission.READ_SMS") plus a 0/1 "label" column
# (1 = malware, 0 = benign).
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "apk_permissions.csv")

PERMISSION_PREFIX = "android.permission."

# Verdict thresholds on the model's malware probability
SAFE_BELOW = 0.35
MALICIOUS_FROM = 0.70
TOP_N_FEATURES = 5

# Permissions that can be useful as indicators during APK analysis.
# These alone do NOT prove that an APK is malicious.
RISKY_PERMISSIONS = {
    "android.permission.READ_SMS",
    "android.permission.RECEIVE_SMS",
    "android.permission.SEND_SMS",
    "android.permission.READ_CONTACTS",
    "android.permission.WRITE_CONTACTS",
    "android.permission.READ_CALL_LOG",
    "android.permission.WRITE_CALL_LOG",
    "android.permission.RECORD_AUDIO",
    "android.permission.CAMERA",
    "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.ACCESS_COARSE_LOCATION",
    "android.permission.READ_PHONE_STATE",
    "android.permission.CALL_PHONE",
    "android.permission.READ_EXTERNAL_STORAGE",
    "android.permission.WRITE_EXTERNAL_STORAGE",
    "android.permission.REQUEST_INSTALL_PACKAGES",
    "android.permission.SYSTEM_ALERT_WINDOW",
    "android.permission.RECEIVE_BOOT_COMPLETED",
}

# Rule weights: still used for the human-readable evidence and as a fallback
PERMISSION_WEIGHTS = {
    "android.permission.READ_SMS": 0.10,
    "android.permission.RECEIVE_SMS": 0.10,
    "android.permission.SEND_SMS": 0.12,
    "android.permission.READ_CONTACTS": 0.05,
    "android.permission.WRITE_CONTACTS": 0.03,
    "android.permission.READ_CALL_LOG": 0.08,
    "android.permission.WRITE_CALL_LOG": 0.05,
    "android.permission.RECORD_AUDIO": 0.05,
    "android.permission.CAMERA": 0.02,
    "android.permission.ACCESS_FINE_LOCATION": 0.02,
    "android.permission.ACCESS_COARSE_LOCATION": 0.01,
    "android.permission.READ_PHONE_STATE": 0.03,
    "android.permission.CALL_PHONE": 0.04,
    "android.permission.READ_EXTERNAL_STORAGE": 0.02,
    "android.permission.WRITE_EXTERNAL_STORAGE": 0.02,
    "android.permission.REQUEST_INSTALL_PACKAGES": 0.10,
    "android.permission.SYSTEM_ALERT_WINDOW": 0.10,
    "android.permission.RECEIVE_BOOT_COMPLETED": 0.03,
}

SMS_READ_PERMISSIONS = {"android.permission.READ_SMS", "android.permission.RECEIVE_SMS"}
SUSPICIOUS_THRESHOLD = 0.30


def _short(name):
    """'android.permission.READ_SMS' -> 'READ_SMS'."""
    name = str(name).strip()
    return name[len(PERMISSION_PREFIX):] if name.startswith(PERMISSION_PREFIX) else name


SYNTHETIC_FEATURES = [_short(p) for p in PERMISSION_WEIGHTS]

# Assumed probability that a benign / malicious app requests each permission.
# These numbers are assumptions, NOT measurements from real apps.
BENIGN_PROBS = {
    "READ_SMS": 0.01, "RECEIVE_SMS": 0.02, "SEND_SMS": 0.01,
    "READ_CONTACTS": 0.15, "WRITE_CONTACTS": 0.05,
    "READ_CALL_LOG": 0.01, "WRITE_CALL_LOG": 0.01,
    "RECORD_AUDIO": 0.08, "CAMERA": 0.25,
    "ACCESS_FINE_LOCATION": 0.20, "ACCESS_COARSE_LOCATION": 0.20,
    "READ_PHONE_STATE": 0.15, "CALL_PHONE": 0.05,
    "READ_EXTERNAL_STORAGE": 0.30, "WRITE_EXTERNAL_STORAGE": 0.30,
    "REQUEST_INSTALL_PACKAGES": 0.02, "SYSTEM_ALERT_WINDOW": 0.03,
    "RECEIVE_BOOT_COMPLETED": 0.10,
}
MALWARE_PROBS = {
    "READ_SMS": 0.60, "RECEIVE_SMS": 0.65, "SEND_SMS": 0.55,
    "READ_CONTACTS": 0.50, "WRITE_CONTACTS": 0.25,
    "READ_CALL_LOG": 0.40, "WRITE_CALL_LOG": 0.20,
    "RECORD_AUDIO": 0.30, "CAMERA": 0.25,
    "ACCESS_FINE_LOCATION": 0.40, "ACCESS_COARSE_LOCATION": 0.35,
    "READ_PHONE_STATE": 0.75, "CALL_PHONE": 0.30,
    "READ_EXTERNAL_STORAGE": 0.50, "WRITE_EXTERNAL_STORAGE": 0.50,
    "REQUEST_INSTALL_PACKAGES": 0.40, "SYSTEM_ALERT_WINDOW": 0.45,
    "RECEIVE_BOOT_COMPLETED": 0.60,
}


def calculate_sha256(file_path):
    """Calculate SHA-256 hash of the APK."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def score_permissions(permissions):
    """
    Rule-based scoring. Returns (risk_score 0-1, verdict, evidence list, contributions list).
    The model's verdict is the final one; this is kept for human-readable evidence.
    """
    perms = set(permissions)
    score = 0.0
    contributions = []
    evidence = []

    # 1) Small weight for every sensitive permission
    found = [p for p in PERMISSION_WEIGHTS if p in perms]
    for p in found:
        score += PERMISSION_WEIGHTS[p]
        contributions.append((_short(p), PERMISSION_WEIGHTS[p]))
    if found:
        evidence.append(f"{len(found)} sensitive permission(s) requested.")

    # 2) Dangerous combinations
    sms_combo = bool(perms & SMS_READ_PERMISSIONS) and "android.permission.SEND_SMS" in perms
    dropper_combo = (
        "android.permission.SYSTEM_ALERT_WINDOW" in perms
        and "android.permission.REQUEST_INSTALL_PACKAGES" in perms
    )
    if sms_combo:
        score += 0.25
        contributions.append(("COMBO: SMS read + send", 0.25))
        evidence.append("Can read and send SMS together - common in OTP-theft / SMS-fraud malware.")
    if dropper_combo:
        score += 0.25
        contributions.append(("COMBO: screen overlay + install apps", 0.25))
        evidence.append("Can draw over other apps and install packages - common in dropper / banking-trojan behaviour.")

    score = min(score, 1.0)

    # 3) Verdict: "malicious" only when both combo families are present
    if sms_combo and dropper_combo:
        verdict = "malicious"
    elif score >= SUSPICIOUS_THRESHOLD:
        verdict = "suspicious"
    else:
        verdict = "safe"

    if not evidence:
        evidence.append("No strong indicators were detected by the permission checks.")

    return round(score, 3), verdict, evidence, contributions


# --------------------------------------------------------------------------
# Model: training, loading, prediction, SHAP
# --------------------------------------------------------------------------

def _generate_synthetic(n=4000, seed=42):
    """Synthetic permission profiles: 60% benign-like, 40% malware-like (assumed, not real)."""
    rng = np.random.default_rng(seed)
    benign = np.array([BENIGN_PROBS[f] for f in SYNTHETIC_FEATURES])
    malware = np.array([MALWARE_PROBS[f] for f in SYNTHETIC_FEATURES])

    labels = rng.random(n) < 0.40
    probs = np.where(labels[:, None], malware[None, :], benign[None, :])
    X = (rng.random((n, len(SYNTHETIC_FEATURES))) < probs).astype(int)

    df = pd.DataFrame(X, columns=SYNTHETIC_FEATURES)
    df["label"] = labels.astype(int)
    return df


def load_training_data():
    """Returns (DataFrame with a 'label' column, source description)."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if "label" not in df.columns:
            raise ValueError("data/apk_permissions.csv needs a 'label' column (1 = malware, 0 = benign).")
        feature_cols = [c for c in df.columns if c != "label"]
        df = df.rename(columns={c: _short(c) for c in feature_cols})
        df = df.loc[:, ~df.columns.duplicated()]
        print(f"Loaded real dataset: {len(df)} APKs, {df.shape[1] - 1} permission columns")
        return df, "real dataset (data/apk_permissions.csv)"

    print("No data/apk_permissions.csv found - using SYNTHETIC permission profiles.")
    return _generate_synthetic(), "synthetic permission profiles (not real malware samples)"


def train():
    df, source = load_training_data()
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=1
    )
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"APK model hold-out accuracy: {acc:.3f}  (source: {source})")
    if source.startswith("synthetic"):
        print("NOTE: this accuracy only measures how well the model learned the synthetic "
              "profiles. It says nothing about real malware.")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {"model": clf, "feature_names": list(X.columns), "training_data": source},
        MODEL_PATH,
    )
    print(f"Model saved to {MODEL_PATH}")
    return clf


def _load_model():
    if not os.path.exists(MODEL_PATH):
        train()
    return joblib.load(MODEL_PATH)


def _feature_row(permissions, feature_names):
    """One-row DataFrame with a 0/1 column per model feature."""
    requested = {_short(p) for p in permissions}
    return pd.DataFrame([{f: int(f in requested) for f in feature_names}])[feature_names]


def _shap_values(clf, X):
    """SHAP values of the 'malware' class for a single-row DataFrame."""
    import shap
    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(X)
    if isinstance(values, list):                 # older shap: one array per class
        return np.asarray(values[1])[0]
    values = np.asarray(values)
    if values.ndim == 3:                         # newer shap: (rows, features, classes)
        return values[0, :, 1]
    return values[0]


def predict_permissions(permissions):
    """
    Runs the model on a permission list.
    Returns (malware_probability, verdict, top_features, explanation_type, training_data).
    """
    bundle = _load_model()
    clf = bundle["model"]
    feature_names = bundle["feature_names"]
    X = _feature_row(permissions, feature_names)

    malware_idx = list(clf.classes_).index(1)
    probability = float(clf.predict_proba(X)[0][malware_idx])

    if probability < SAFE_BELOW:
        verdict = "safe"
    elif probability < MALICIOUS_FROM:
        verdict = "suspicious"
    else:
        verdict = "malicious"

    # Only permissions the APK actually requests are shown as factors
    try:
        values = _shap_values(clf, X)
        pairs = [
            (name, float(v))
            for name, v, present in zip(feature_names, values, X.iloc[0].tolist())
            if present
        ]
        explanation_type = "shap"
    except Exception as exc:
        # SHAP failed (e.g. not installed): fall back to the rule weights
        print(f"[APK Agent] SHAP unavailable, using rule weights instead: {exc}")
        _, _, _, heuristic = score_permissions(permissions)
        pairs = [(name, float(w)) for name, w in heuristic]
        explanation_type = "heuristic"

    pairs.sort(key=lambda item: -abs(item[1]))
    top_features = [[name, round(value, 4)] for name, value in pairs[:TOP_N_FEATURES]]

    return probability, verdict, top_features, explanation_type, bundle["training_data"]


def analyze_apk(apk_path):
    """
    Analyze an Android APK and return a structured CyberLens result.
    """

    if not os.path.isfile(apk_path):
        raise FileNotFoundError(f"APK file not found: {apk_path}")

    if not apk_path.lower().endswith(".apk"):
        raise ValueError("The supplied file is not an APK.")

    print(f"[APK Agent] Analyzing: {apk_path}")

    # Androguard analysis
    apk, dex_objects, analysis = AnalyzeAPK(apk_path)

    # ---------------------------------------------------------
    # Basic APK information
    # ---------------------------------------------------------

    package_name = apk.get_package()

    try:
        app_name = apk.get_app_name()
    except Exception:
        app_name = None

    # ---------------------------------------------------------
    # Permissions
    # ---------------------------------------------------------

    permissions = []

    try:
        permissions = apk.get_permissions() or []
    except Exception:
        permissions = []

    permissions = sorted(set(permissions))

    risky_permissions = [
        permission
        for permission in permissions
        if permission in RISKY_PERMISSIONS
    ]

    # ---------------------------------------------------------
    # DEX / code information
    # ---------------------------------------------------------

    class_names = []
    method_names = []

    for dex in dex_objects:
        try:
            classes = dex.get_classes()

            for clazz in classes:
                class_name = clazz.get_name()

                if class_name:
                    class_names.append(class_name)

                try:
                    for method in clazz.get_methods():
                        method_name = method.get_name()

                        if method_name:
                            method_names.append(method_name)

                except Exception:
                    pass

        except Exception:
            pass

    class_names = sorted(set(class_names))
    method_names = sorted(set(method_names))

    # ---------------------------------------------------------
    # APK hash
    # ---------------------------------------------------------

    sha256 = calculate_sha256(apk_path)

    # ---------------------------------------------------------
    # Model prediction + SHAP, plus rule-based evidence
    # ---------------------------------------------------------

    probability, verdict, top_features, explanation_type, model_source = \
        predict_permissions(permissions)
    heuristic_score, heuristic_verdict, evidence, heuristic_contributions = \
        score_permissions(permissions)

    risky = verdict in ("suspicious", "malicious")
    confidence = probability if risky else 1.0 - probability

    result = {
        "agent": "apk",
        "input_type": "android_apk",
        "apk_path": os.path.abspath(apk_path),

        "package_name": package_name,
        "app_name": app_name,

        "verdict": verdict,
        "risk_score": round(probability, 3),       # model malware probability, 0-1
        "confidence": round(confidence, 3),        # probability of the predicted verdict
        "top_features": top_features,              # [[permission, shap_value], ...]
        "explanation_type": explanation_type,      # "shap" or "heuristic" (fallback)
        "model_source": model_source,

        "heuristic_score": heuristic_score,
        "heuristic_verdict": heuristic_verdict,
        "heuristic_contributions": heuristic_contributions,

        "permissions": permissions,
        "risky_permissions": risky_permissions,

        "class_count": len(class_names),
        "method_count": len(method_names),

        # Keep the result manageable.
        "classes": class_names[:100],
        "methods": method_names[:100],

        "sha256": sha256,

        "evidence": evidence,
    }

    return result


def print_result(result):
    """Print the APK analysis result in a readable format."""

    print("\n" + "=" * 60)
    print("CYBERLENS APK ANALYSIS RESULT")
    print("=" * 60)

    print(f"APK           : {result['apk_path']}")
    print(f"App Name      : {result['app_name']}")
    print(f"Package       : {result['package_name']}")

    print(f"\nVerdict       : {result['verdict'].upper()}")
    print(f"Malware prob. : {result['risk_score']}")
    print(f"Confidence    : {result['confidence']}")
    print(f"Explanation   : {result['explanation_type']}")
    print(f"Model trained : {result['model_source']}")
    print(f"Rule-based    : {result['heuristic_verdict']} ({result['heuristic_score']})")

    print(f"\nPermissions   : {len(result['permissions'])}")
    print(f"Risky Perms   : {len(result['risky_permissions'])}")

    print(f"Classes       : {result['class_count']}")
    print(f"Methods       : {result['method_count']}")

    print(f"\nSHA-256:")
    print(result["sha256"])

    print("\nRisky Permissions:")

    if result["risky_permissions"]:
        for permission in result["risky_permissions"]:
            print(f"  - {permission}")
    else:
        print("  None detected")

    print("\nTop Factors (SHAP value, + raises risk, - lowers it):")

    if result["top_features"]:
        for name, value in result["top_features"]:
            print(f"  - {name}: {value:+.4f}")
    else:
        print("  None (no sensitive permissions requested)")

    print("\nEvidence:")

    for item in result["evidence"]:
        print(f"  - {item}")

    print("=" * 60)


def main():
    """Command-line entry point."""

    if len(sys.argv) == 2 and sys.argv[1] == "--train":
        train()
        return

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            'python -m agents.apk_agent "path\\to\\file.apk"\n'
            "python -m agents.apk_agent --train"
        )
        return

    apk_path = sys.argv[1]

    try:
        result = analyze_apk(apk_path)
        print_result(result)

    except Exception as error:
        print("\n[APK Agent ERROR]")
        print(str(error))


if __name__ == "__main__":
    main()