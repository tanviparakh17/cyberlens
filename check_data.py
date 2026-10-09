import sys
sys.path.insert(0, ".")
import pandas as pd
from sklearn.metrics import roc_auc_score
from utils.feature_extraction import extract_url_features

d = pd.read_csv("data/phishing_urls.csv")
d["len"] = d["url"].str.len()
print(d.groupby("label")["len"].describe().round(1))

X = pd.DataFrame([extract_url_features(u) for u in d["url"]])
rows = []
for c in X.columns:
    a = roc_auc_score(d["label"], X[c])
    rows.append((max(a, 1 - a), c))
print("\nSingle-feature AUC (0.97+ = shortcut ka khatra):")
for a, c in sorted(rows, reverse=True)[:6]:
    print(f"  {c:30s} {a:.3f}")