import pandas as pd
import os

print("Loading phishing dataset...")
phishing_df = pd.read_csv("data/phishing_site_urls.csv")

# Is dataset mein columns "URL" aur "Label" hain, "Label" mein "bad"/"good" values hain
phishing_df = phishing_df.rename(columns={"URL": "url", "Label": "label"})

# Sirf "bad" (phishing) wale URLs lene hain is file se
phishing_only = phishing_df[phishing_df["label"] == "bad"][["url"]].copy()
phishing_only["label"] = 1   # 1 = phishing

# ---- FIX: URLs ko normalize karo -- raw Kaggle URLs mein http/https
# prefix nahi hota, jabki humare safe URLs mein hamesha hota hai.
# Bina isko fix kiye, "scheme hai ya nahi" khud ek fake signal ban jaata hai
# (num_redirect_chars feature ismein accidentally leak ho raha tha)
phishing_only["url"] = phishing_only["url"].apply(
    lambda u: u if u.startswith(("http://", "https://")) else "http://" + u
)

print(f"Phishing URLs found: {len(phishing_only)}")

print("Loading Tranco top sites (legitimate domains)...")
# Tranco list mein header nahi hota, format: rank,domain
tranco_df = pd.read_csv("data/top-1m.csv", header=None, names=["rank", "domain"])

# Poori 1M list se random sample lo (sirf top-ranked famous sites nahi),
# taaki chhoti/lesser-known legitimate sites bhi represent hon
tranco_df = tranco_df.sample(n=50000, random_state=42)

safe_only = pd.DataFrame({
    "url": "https://" + tranco_df["domain"].astype(str),
    "label": 0   # 0 = safe
})

print(f"Safe URLs found: {len(safe_only)}")

# ---- HARD EXAMPLES: dataset ko real-world jaisa challenging banane ke liye ----

# Hard NEGATIVES: legitimate URLs jo "phishing jaisi" dikhti hain (lambi, complex)
hard_safe_urls = pd.DataFrame({
    "url": [
        "https://accounts.google.com/signin/v2/identifier?service=mail&continue=https",
        "https://www.amazon.com/gp/aw/d/B08N5WRWNW/ref=sspa_dk_detail_0",
        "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "https://www.paypal.com/signin/?returnUri=https://www.paypal.com/myaccount",
        "https://secure.bankofamerica.com/login/sign-in/signOnV2Screen.go",
    ],
    "label": 0
})

# Hard POSITIVES: phishing URLs jo "safe jaisi" dikhti hain (chotی, clean)
hard_phishing_urls = pd.DataFrame({
    "url": [
        "http://amaz0n-verify.com",
        "http://paypa1.com",
        "http://faceb00k-login.com",
        "http://secure-hdfc.in",
        "http://netfIix.com",
    ],
    "label": 1
})

# ---- Balanced sample banao ----
SAMPLE_SIZE_PER_CLASS = 2000

phishing_sample = phishing_only.sample(
    n=min(len(phishing_only), SAMPLE_SIZE_PER_CLASS), random_state=42
)
safe_sample = safe_only.sample(
    n=min(len(safe_only), SAMPLE_SIZE_PER_CLASS), random_state=42
)

# ---- Combine aur shuffle karo ----
final_df = pd.concat(
    [phishing_sample, safe_sample, hard_safe_urls, hard_phishing_urls],
    ignore_index=True
)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ---- Save final file ----
final_df.to_csv("data/phishing_urls.csv", index=False)

print(f"\n✅ Final dataset saved to data/phishing_urls.csv")
print(f"   Total: {len(final_df)} URLs")
print(f"   Phishing: {(final_df['label'] == 1).sum()}")
print(f"   Safe: {(final_df['label'] == 0).sum()}")