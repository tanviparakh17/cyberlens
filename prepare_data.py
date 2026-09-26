import pandas as pd
import os

print("Loading phishing dataset...")
phishing_df = pd.read_csv("data/phishing_site_urls.csv")

# Is dataset mein columns "URL" aur "Label" hain, "Label" mein "bad"/"good" values hain
phishing_df = phishing_df.rename(columns={"URL": "url", "Label": "label"})

# Sirf "bad" (phishing) wale URLs lene hain is file se
phishing_only = phishing_df[phishing_df["label"] == "bad"][["url"]].copy()
phishing_only["label"] = 1   # 1 = phishing

print(f"Phishing URLs found: {len(phishing_only)}")

print("Loading Tranco top sites (legitimate domains)...")
# Tranco list mein header nahi hota, format: rank,domain
tranco_df = pd.read_csv("data/top-1m.csv", header=None, names=["rank", "domain"])

safe_only = pd.DataFrame({
    "url": "https://" + tranco_df["domain"].astype(str),
    "label": 0   # 0 = safe
})

print(f"Safe URLs found: {len(safe_only)}")

# ---- Balanced sample banao ----
SAMPLE_SIZE_PER_CLASS = 2000

phishing_sample = phishing_only.sample(
    n=min(len(phishing_only), SAMPLE_SIZE_PER_CLASS), random_state=42
)
safe_sample = safe_only.sample(
    n=min(len(safe_only), SAMPLE_SIZE_PER_CLASS), random_state=42
)

# ---- Combine aur shuffle karo ----
final_df = pd.concat([phishing_sample, safe_sample], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ---- Save final file ----
final_df.to_csv("data/phishing_urls.csv", index=False)

print(f"\n✅ Final dataset saved to data/phishing_urls.csv")
print(f"   Total: {len(final_df)} URLs")
print(f"   Phishing: {(final_df['label'] == 1).sum()}")
print(f"   Safe: {(final_df['label'] == 0).sum()}")