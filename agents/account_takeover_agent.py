"""
agents/account_takeover_agent.py
================================
CyberLens - Account Takeover (ATO) Detection Agent (graph-based)

Idea in one paragraph
---------------------
When an account gets hijacked, the attacker usually uses it to message lots of
people quickly: the same scam text, sent in a burst, often to people the real
owner never talked to, and few of them reply. We model the message log as a
directed graph with networkx (node = account, edge A->B = "A messaged B", with
first/last timestamps and a message count on the edge), compare each account's
RECENT behaviour (the observation window, last 7 days) against its HISTORY, and
let a RandomForest decide "suspicious" vs "normal".

Interface (same pattern as url_agent / email_agent / qr_agent)
--------------------------------------------------------------
    train()             -> trains on data/ato_logs.csv + data/ato_labels.csv,
                           saves models/account_takeover_model.joblib
    predict(input_data) -> dict with "account_id", "message_logs", "verdict",
                           "confidence", "features", "model", "feature_row"
                           (plus "risk_score" and "agent")

    input_data = {
        "account_id":   "user_0042",                       # account to judge
        "message_logs": <list of dicts | DataFrame | CSV path>
                        # columns: timestamp, sender, recipient, message
                        # MUST include the account's history, not just the
                        # suspicious burst, because features compare recent vs past
        "end_time":     "2026-10-01T00:00:00",              # optional; default = last log timestamp
    }

Features (one row per account, all computed for the last OBS_WINDOW_DAYS)
------------------------------------------------------------------------
    msgs_sent                 outgoing messages in the window
    distinct_recipients       out-degree of the account in the window graph
    outdegree_burst_rate      max distinct recipients contacted in any 10-minute window
    burst_vs_baseline         window daily send rate / (historical daily rate + 1)
    content_repetition_score  share of sent messages whose exact text went to >= 3 different people
    max_identical_recipients  most recipients that got one identical text
    unfamiliar_contact_ratio  share of recipients this account contacted for the first time ever
                              (no earlier message in EITHER direction)
    reply_ratio               share of recipients who wrote back after being messaged
    link_ratio                share of sent messages containing a URL
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ato_logs.csv"
LABELS_PATH = BASE_DIR / "data" / "ato_labels.csv"
MODEL_PATH = BASE_DIR / "models" / "account_takeover_model.joblib"

OBS_WINDOW_DAYS = 7          # "recent behaviour" window; everything before it is history
BURST_WINDOW_MINUTES = 10    # sliding window used for the burst feature
REPEAT_MIN_RECIPIENTS = 3    # identical text to >= this many people counts as repetition
REPEAT_MIN_CHARS = 15        # ignore short phatic texts ("ok", "thanks") for repetition
SUSPICION_THRESHOLD = 0.5    # P(compromised) at or above this -> "suspicious"
SEPARATION_AUC_ALERT = 0.97  # single-feature AUC above this is flagged as "too clean"

REQUIRED_COLUMNS = ["timestamp", "sender", "recipient", "message"]
FEATURE_NAMES = [
    "msgs_sent",
    "distinct_recipients",
    "outdegree_burst_rate",
    "burst_vs_baseline",
    "content_repetition_score",
    "max_identical_recipients",
    "unfamiliar_contact_ratio",
    "reply_ratio",
    "link_ratio",
]
URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)

_MODEL_CACHE: dict | None = None  # predict() loads the model once, then reuses it


# ===========================================================================
# PART 1 - FEATURE EXTRACTION (no ML here; reusable on its own)
# ===========================================================================
def load_logs(source) -> pd.DataFrame:
    """Accept a CSV path, a DataFrame, or a list of dicts; return a clean, time-sorted DataFrame."""
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.DataFrame(list(source or []))

    if df.empty:
        return pd.DataFrame({
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "sender": pd.Series(dtype=str),
            "recipient": pd.Series(dtype=str),
            "message": pd.Series(dtype=str),
        })

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"message_logs is missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["sender"] = df["sender"].astype(str)
    df["recipient"] = df["recipient"].astype(str)
    df["message"] = df["message"].fillna("").astype(str)
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def build_interaction_graph(logs: pd.DataFrame) -> nx.DiGraph:
    """
    Directed graph: node = account, edge A->B = A sent at least one message to B.
    Edge attributes: n_msgs, first_ts, last_ts.
    """
    graph = nx.DiGraph()
    if logs.empty:
        return graph
    edges = (
        logs.groupby(["sender", "recipient"])["timestamp"]
        .agg(n_msgs="count", first_ts="min", last_ts="max")
        .reset_index()
    )
    graph.add_edges_from(
        (row.sender, row.recipient, {"n_msgs": int(row.n_msgs), "first_ts": row.first_ts, "last_ts": row.last_ts})
        for row in edges.itertuples(index=False)
    )
    return graph


def _normalise_text(text: str) -> str:
    """Lower-case and collapse whitespace so trivially re-formatted copies still match."""
    return " ".join(str(text).lower().split())


def _max_distinct_in_window(timestamps_ns: np.ndarray, recipients: list, window_ns: int) -> int:
    """Two-pointer sliding window: most DISTINCT recipients contacted within `window_ns`."""
    counts, left, best = Counter(), 0, 0
    for right in range(len(timestamps_ns)):
        counts[recipients[right]] += 1
        while timestamps_ns[right] - timestamps_ns[left] > window_ns:
            counts[recipients[left]] -= 1
            if counts[recipients[left]] == 0:
                del counts[recipients[left]]
            left += 1
        best = max(best, len(counts))
    return best


def _features_for_one_account(account, sent, g_all, g_obs, obs_start, hist_sent_count, first_seen) -> dict:
    """Compute the feature dict for a single account (see module docstring for definitions)."""
    features = dict.fromkeys(FEATURE_NAMES, 0.0)
    if sent is None or sent.empty:
        return features  # sent nothing recently -> nothing suspicious to measure

    n_sent = len(sent)
    recipients = sent["recipient"].tolist()
    distinct = sorted(set(recipients))
    features["msgs_sent"] = float(n_sent)
    features["distinct_recipients"] = float(len(distinct))

    # --- Outdegree burst rate -------------------------------------------------
    ts_ns = sent["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64")
    window_ns = int(pd.Timedelta(minutes=BURST_WINDOW_MINUTES).value)
    features["outdegree_burst_rate"] = float(_max_distinct_in_window(ts_ns, recipients, window_ns))

    # --- Recent send rate compared with the account's own past ------------------
    if first_seen is not None and first_seen < obs_start:
        history_days = max(1.0, (obs_start - first_seen) / pd.Timedelta(days=1))
        hist_rate = hist_sent_count / history_days
    else:
        hist_rate = 0.0  # brand-new account: no baseline yet
    features["burst_vs_baseline"] = (n_sent / OBS_WINDOW_DAYS) / (hist_rate + 1.0)

    # --- Content repetition -----------------------------------------------------
    texts = sent["message"].map(_normalise_text)
    long_mask = texts.str.len() >= REPEAT_MIN_CHARS
    if long_mask.any():
        recipients_per_text = sent.loc[long_mask].groupby(texts[long_mask])["recipient"].nunique()
        repeated_texts = set(recipients_per_text[recipients_per_text >= REPEAT_MIN_RECIPIENTS].index)
        features["content_repetition_score"] = float(texts.isin(repeated_texts).sum()) / n_sent
        features["max_identical_recipients"] = float(recipients_per_text.max())

    # --- Unfamiliar contacts and replies (read straight off the graphs) ----------
    unfamiliar, replied = 0, 0
    for r in distinct:
        out_edge = g_all[account][r]
        # This account started the relationship, and it started inside the window
        # -> no earlier interaction in either direction -> unfamiliar contact.
        back_edge = g_all.get_edge_data(r, account)
        if out_edge["first_ts"] > obs_start and (back_edge is None or back_edge["first_ts"] > out_edge["first_ts"]):
            unfamiliar += 1
        # Did r write back after this account messaged them in the window?
        obs_back = g_obs.get_edge_data(r, account)
        if obs_back is not None and obs_back["last_ts"] > g_obs[account][r]["first_ts"]:
            replied += 1
    features["unfamiliar_contact_ratio"] = unfamiliar / len(distinct)
    features["reply_ratio"] = replied / len(distinct)

    # --- Links ------------------------------------------------------------------
    features["link_ratio"] = float(sent["message"].str.contains(URL_PATTERN).sum()) / n_sent
    return features


def extract_features_for_accounts(message_logs, account_ids, end_time=None) -> pd.DataFrame:
    """
    Build the graphs ONCE and compute features for many accounts.
    Returns a DataFrame indexed by account_id with columns == FEATURE_NAMES.
    """
    logs = load_logs(message_logs)
    if end_time is None:
        end_time = logs["timestamp"].max() if not logs.empty else pd.Timestamp.now()
    end_time = pd.Timestamp(end_time)
    obs_start = end_time - pd.Timedelta(days=OBS_WINDOW_DAYS)

    logs = logs[logs["timestamp"] <= end_time]              # ignore anything "from the future"
    obs = logs[logs["timestamp"] > obs_start]                # observation window
    hist = logs[logs["timestamp"] <= obs_start]              # history

    g_all = build_interaction_graph(logs)   # full history up to end_time (for familiarity)
    g_obs = build_interaction_graph(obs)    # window only (for replies)

    sent_by_account = dict(tuple(obs.groupby("sender")))
    hist_sent_counts = hist.groupby("sender").size()
    first_seen = pd.concat([
        logs[["sender", "timestamp"]].rename(columns={"sender": "account"}),
        logs[["recipient", "timestamp"]].rename(columns={"recipient": "account"}),
    ]).groupby("account")["timestamp"].min()

    rows = {}
    for account in account_ids:
        account = str(account)
        rows[account] = _features_for_one_account(
            account=account,
            sent=sent_by_account.get(account),
            g_all=g_all,
            g_obs=g_obs,
            obs_start=obs_start,
            hist_sent_count=int(hist_sent_counts.get(account, 0)),
            first_seen=first_seen.get(account),
        )
    X = pd.DataFrame.from_dict(rows, orient="index")[FEATURE_NAMES].astype(float)
    X.index.name = "account_id"
    return X


def extract_features(account_id, message_logs, end_time=None) -> dict:
    """Convenience wrapper: features for a single account as a plain dict."""
    return extract_features_for_accounts(message_logs, [account_id], end_time).iloc[0].to_dict()


# ===========================================================================
# PART 2 - SANITY CHECK FOR "TOO GOOD TO BE TRUE" FEATURES (lesson from url_agent)
# ===========================================================================
def audit_feature_separation(X: pd.DataFrame, y: pd.Series) -> list:
    """
    Print per-class mean/std for every feature and flag features that separate the
    classes almost perfectly on their own. A flag doesn't prove a bug, but it means
    "go check whether this is a real signal or a quirk of how the data was made".
    """
    df = X.copy()
    df["label"] = y.to_numpy()
    print("\n=== Feature audit: per-class mean (0 = normal, 1 = compromised) ===")
    print(df.groupby("label").mean().T.round(3).to_string())
    print("\n=== Feature audit: per-class std ===")
    print(df.groupby("label").std().T.round(3).to_string())

    print(f"\n=== Single-feature separation (AUC; flag if >= {SEPARATION_AUC_ALERT}) ===")
    flagged = []
    for col in X.columns:
        auc = roc_auc_score(y, X[col])
        separation = max(auc, 1 - auc)  # direction doesn't matter
        normal_vals, comp_vals = X.loc[y.to_numpy() == 0, col], X.loc[y.to_numpy() == 1, col]
        no_overlap = normal_vals.max() < comp_vals.min() or comp_vals.max() < normal_vals.min()
        flag = separation >= SEPARATION_AUC_ALERT or no_overlap
        if flag:
            flagged.append(col)
        note = "  <-- SUSPICIOUSLY CLEAN, investigate" if flag else ""
        print(f"  {col:<26} AUC={separation:.3f}{note}")
    if not flagged:
        print("  OK: no single feature separates the classes on its own.")
    return flagged


# ===========================================================================
# PART 3 - TRAINING
# ===========================================================================
def train(data_path=DATA_PATH, labels_path=LABELS_PATH, model_path=MODEL_PATH, random_state=42) -> dict:
    """Extract features for every labelled account, train a RandomForest, evaluate, save."""
    print(f"Loading logs from {data_path}")
    logs = load_logs(data_path)
    labels = pd.read_csv(labels_path)
    print(f"  {len(logs):,} messages, {len(labels):,} labelled accounts")

    print("Extracting graph features...")
    X = extract_features_for_accounts(logs, labels["account_id"])
    y = labels.set_index("account_id").loc[X.index, "label"].astype(int)
    scenario = labels.set_index("account_id").loc[X.index, "scenario"] if "scenario" in labels else None

    flagged = audit_feature_separation(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,          # slightly smoother trees, less over-fitting on 800 rows
        class_weight="balanced",     # compromised accounts are the minority class
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # --- Evaluation -------------------------------------------------------------
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    print("\n=== Test-set evaluation ===")
    print(f"Accuracy: {accuracy:.4f}   ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["normal", "compromised"], digits=3))
    print("Confusion matrix [rows = true normal/compromised, cols = predicted]:")
    print(confusion_matrix(y_test, y_pred))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    if scenario is not None:
        # Which behaviours does the model get wrong? (scenario is NOT a feature)
        per_scenario = pd.DataFrame({"scenario": scenario.loc[X_test.index], "correct": y_pred == y_test.to_numpy()})
        print("\nTest accuracy per simulated scenario (hard cases: broadcaster, networker, friend_scam, slow_low):")
        print(per_scenario.groupby("scenario")["correct"].agg(["mean", "count"]).round(3).to_string())

    print("\nFeature importances:")
    for name, imp in sorted(zip(X.columns, clf.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:<26} {imp:.3f}")

    # --- Save (same bundle format as the other agents) ---------------------------
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_names": list(X.columns)}, model_path)
    print(f"\nModel saved -> {model_path}")

    global _MODEL_CACHE
    _MODEL_CACHE = None  # force predict() to reload the fresh model

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "flagged_features": flagged,
        "model_path": str(model_path),
    }


# ===========================================================================
# PART 4 - PREDICTION
# ===========================================================================
def _load_model(model_path=MODEL_PATH) -> dict:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"No model at {model_path}. Run train() first.")
        _MODEL_CACHE = joblib.load(model_path)
    return _MODEL_CACHE


def predict(input_data: dict) -> dict:
    """
    Judge one account. See the module docstring for the input_data format.
    Returns the standard CyberLens agent dict (works with xai_agent and report_agent).
    """
    if not isinstance(input_data, dict) or "account_id" not in input_data or "message_logs" not in input_data:
        raise ValueError('input_data must be a dict with "account_id" and "message_logs"')

    bundle = _load_model()
    clf, feature_names = bundle["model"], bundle["feature_names"]

    account_id = str(input_data["account_id"])
    X = extract_features_for_accounts(input_data["message_logs"], [account_id], input_data.get("end_time"))
    feature_row = X[feature_names].reset_index(drop=True)  # exact 1-row vector the model sees

    risk = float(clf.predict_proba(feature_row)[0][list(clf.classes_).index(1)])
    is_suspicious = risk >= SUSPICION_THRESHOLD

    return {
        "agent": "account_takeover",
        "account_id": account_id,
        "message_logs": input_data["message_logs"],                 # raw input, untouched
        "verdict": "suspicious" if is_suspicious else "normal",
        "confidence": round(risk if is_suspicious else 1.0 - risk, 4),  # confidence in the verdict
        "risk_score": round(risk, 4),                               # P(compromised), handy for the UI
        "features": {k: round(float(v), 4) for k, v in feature_row.iloc[0].items()},
        "model": clf,                                               # for xai_agent's SHAP TreeExplainer
        "feature_row": feature_row,
    }


# ===========================================================================
# PART 5 - DEMO / SMOKE TEST
# ===========================================================================
def _build_demo_logs(scenario: str, seed: int = 7) -> list:
    """
    Hand-built logs for one account "demo_user" with 20 friends: 30 days of ordinary
    chatting, plus one scenario injected into the last 7 days. These accounts are NOT
    in the training data, so this is a genuine out-of-sample sanity check.
    """
    import random
    rng = random.Random(seed)
    start = pd.Timestamp("2026-09-01")
    me, friends = "demo_user", [f"friend_{i:02d}" for i in range(20)]
    openers = ["hey", "hi", "yo", "morning"]
    topics = ["are we still on for lunch", "did you finish the assignment", "call me when you're free",
              "running 10 min late", "send me the photos", "movie at 9?", "how was the interview",
              "can we move the meeting", "thanks for yesterday", "who's driving tonight"]
    logs = []

    def add(ts, sender, recipient, message):
        logs.append({"timestamp": ts.isoformat(), "sender": sender, "recipient": recipient, "message": message})

    def chat():
        return rng.choice(["ok", "lol", "sure", "thanks"]) if rng.random() < 0.3 else f"{rng.choice(openers)} {rng.choice(topics)}"

    # 30 days of normal life (the owner keeps chatting even if hacked)
    for day in range(30):
        for _ in range(rng.randint(2, 6)):
            ts = start + pd.Timedelta(days=day, hours=rng.randint(9, 22), minutes=rng.randint(0, 59))
            friend = rng.choice(friends)
            add(ts, me, friend, chat())
            if rng.random() < 0.55:
                add(ts + pd.Timedelta(minutes=rng.randint(1, 60)), friend, me, chat())

    obs_day = start + pd.Timedelta(days=26)
    if scenario == "party_invite":       # NORMAL, but looks like a burst (hard negative)
        t = obs_day + pd.Timedelta(hours=19)
        for f in friends[:18]:
            t += pd.Timedelta(seconds=rng.randint(5, 15))
            add(t, me, f, "hey everyone! housewarming party at my place saturday 8pm https://forms.gle/aB3dE9xY")
            if rng.random() < 0.7:
                add(t + pd.Timedelta(minutes=rng.randint(2, 40)), f, me, "count me in!")
    elif scenario == "friend_money_scam":  # COMPROMISED, but only messages existing friends
        t = obs_day + pd.Timedelta(hours=3, minutes=14)
        for f in friends:
            t += pd.Timedelta(seconds=rng.randint(10, 40))
            add(t, me, f, "Hey, I'm stuck abroad and lost my wallet, can you send me some money urgently? I'll pay you back tomorrow")
            if rng.random() < 0.25:
                add(t + pd.Timedelta(minutes=rng.randint(2, 30)), f, me, "wait is this really you??")
    elif scenario == "slow_low_attacker":  # COMPROMISED, no burst, personalised texts
        names = ["Priya", "Rahul", "Sneha", "Arjun", "Meera", "Kabir", "Isha"]
        for i in range(14):
            ts = start + pd.Timedelta(days=24 + i // 3, hours=rng.randint(0, 23), minutes=rng.randint(0, 59))
            add(ts, me, f"stranger_{i:02d}", f"Hi {names[i % len(names)]}, your parcel delivery failed, please reschedule here http://bit.ly/x{i}Qz")
        add(start + pd.Timedelta(days=28), "stranger_03", me, "who is this?")
    # scenario == "quiet_normal": nothing extra

    return logs


if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    metrics = train()

    print("\n" + "=" * 70)
    print("PREDICTION SANITY CHECKS (out-of-sample, hand-built accounts)")
    print("=" * 70)
    demo_end = "2026-10-01T00:00:00"
    cases = [
        ("quiet_normal", "normal", "ordinary month of chatting"),
        ("party_invite", "normal", "HARD: same invite + link to 18 friends in 4 min"),
        ("friend_money_scam", "suspicious", "HARD: only existing friends, no link, 3am"),
        ("slow_low_attacker", "suspicious", "HARD: no burst, personalised texts to strangers"),
    ]
    for scenario, expected, description in cases:
        result = predict({"account_id": "demo_user", "message_logs": _build_demo_logs(scenario), "end_time": demo_end})
        status = "PASS" if result["verdict"] == expected else "FAIL"
        print(f"\n[{status}] {scenario}: {description}")
        print(f"  verdict={result['verdict']}  confidence={result['confidence']:.3f}  "
              f"risk_score={result['risk_score']:.3f}  (expected {expected})")
        print(f"  features={result['features']}")

    # Edge case: account that isn't in the logs at all -> must not crash
    ghost = predict({"account_id": "ghost_account", "message_logs": _build_demo_logs("quiet_normal"), "end_time": demo_end})
    print(f"\n[EDGE] unknown account -> verdict={ghost['verdict']} confidence={ghost['confidence']:.3f}")

    # Contract check: the keys xai_agent / report_agent rely on
    required = {"verdict", "confidence", "features", "model", "feature_row", "message_logs"}
    assert required <= set(ghost), f"missing keys: {required - set(ghost)}"
    assert ghost["feature_row"].shape == (1, len(FEATURE_NAMES))
    assert 0.0 <= ghost["confidence"] <= 1.0
    print("\nInterface contract OK: all required keys present, feature_row is 1 x", len(FEATURE_NAMES))
