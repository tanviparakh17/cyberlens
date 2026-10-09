import pandas as pd
import random
import os

# Fixed seed taaki har baar same dataset bane (result reproducible rahe)
rng = random.Random(42)

print("Loading phishing dataset...")
phishing_df = pd.read_csv("data/phishing_site_urls.csv")

# Is dataset mein columns "URL" aur "Label" hain, "Label" mein "bad"/"good" values hain
phishing_df = phishing_df.rename(columns={"URL": "url", "Label": "label"})

# Sirf "bad" (phishing) wale URLs lene hain is file se
phishing_only = phishing_df[phishing_df["label"] == "bad"][["url"]].copy()
phishing_only["label"] = 1   # 1 = phishing

# Raw Kaggle URLs mein http/https prefix nahi hota -- consistency ke liye add karo
phishing_only["url"] = phishing_only["url"].apply(
    lambda u: u if u.startswith(("http://", "https://")) else "http://" + u
)

print(f"Phishing URLs found: {len(phishing_only)}")

print("Loading Tranco top sites (legitimate domains)...")
# Tranco list mein header nahi hota, format: rank,domain
tranco_df = pd.read_csv("data/top-1m.csv", header=None, names=["rank", "domain"])

# Poori 1M list se random sample lo (sirf top-ranked famous sites nahi)
tranco_df = tranco_df.sample(n=50000, random_state=42)


# ---- FIX: safe URLs ko realistic banao ----
# Pehle safe URLs sirf "https://domain.com" hote the (na www., na path), jabki asli
# websites ke links mein www., subdomain aur path normal hote hain. Isse model ne
# galat seekh liya tha ki "subdomain ya path hai = phishing" (www.wikipedia.org bhi
# phishing dikh raha tha). Ab har safe domain ko asli jaisa structure dete hain.
PLAIN_PATHS = [
    "/", "/about", "/contact", "/products", "/products/12345",
    "/blog/how-to-get-started", "/careers", "/help/faq", "/docs/getting-started",
    "/en/pricing", "/news/2024/05/update", "/search?q=shoes", "/store/item/48213",
]
# Kuch asli safe URLs mein "login"/"account" jaise words hote hain -- thode se
# examples rakhte hain taaki ye words akele phishing ka shortcut na ban jaayein
KEYWORD_PATHS = ["/login", "/signin?next=/dashboard", "/account/settings", "/secure/checkout"]
OTHER_SUBDOMAINS = ["blog.", "docs.", "support.", "api.", "shop."]


def make_safe_url(domain: str) -> str:
    r = rng.random()
    if r < 0.55:
        sub = ""                               # 55%: seedha domain
    elif r < 0.85:
        sub = "www."                           # 30%: www.
    else:
        sub = rng.choice(OTHER_SUBDOMAINS)     # 15%: blog., docs. jaisa subdomain

    r = rng.random()
    if r < 0.40:
        path = ""                              # 40%: koi path nahi
    elif r < 0.94:
        path = rng.choice(PLAIN_PATHS)         # 54%: normal path
    else:
        path = rng.choice(KEYWORD_PATHS)       # 6%: login/account jaisa path
    return "https://" + sub + domain + path


# ---- SAFE URLs: asli "good" URLs (Kaggle) + thode modern domains (Tranco) ----
# Pehle sab safe URLs Tranco domains se khud bane the aur bahut chhote the (max 76 chars),
# jabki asli good URLs 1000+ chars tak jaate hain. Isse model ne "lamba URL = phishing"
# seekh liya. Ab zyada tar safe URLs wahi source se lete hain jahan se phishing aaye
# (same era, asli paths), aur Tranco wale modern domains thode se milate hain.
N_REAL_SAFE = 1400
N_SYNTH_SAFE = 600

good_df = phishing_df[phishing_df["label"] == "good"][["url"]].copy()
good_df["url"] = good_df["url"].apply(
    lambda u: u if u.startswith(("http://", "https://")) else "http://" + u
)
good_df["host"] = (good_df["url"].str.replace(r"^https?://", "", regex=True)
                   .str.split("/").str[0].str.lower())
good_df = good_df[good_df["host"].str.contains(".", regex=False)]   # junk rows hatao
# Ek domain se zyada se zyada 2 URLs, taaki 1-2 badi sites poora data na bhar dein
good_df = (good_df.sample(frac=1, random_state=42)
           .groupby("host").head(2).drop(columns="host"))
real_safe = good_df.sample(n=min(len(good_df), N_REAL_SAFE), random_state=42).copy()
real_safe["label"] = 0

synth_safe = pd.DataFrame({
    "url": [make_safe_url(d) for d in tranco_df["domain"].astype(str).head(N_SYNTH_SAFE)],
    "label": 0
})
safe_only = pd.concat([real_safe, synth_safe], ignore_index=True)


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

# Hard POSITIVES: phishing URLs jo "safe jaisi" dikhti hain (chhoti, clean)
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