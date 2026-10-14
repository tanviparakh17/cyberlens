import random

import numpy as np
import pandas as pd

# Fixed seed so the dataset is the same every time it is built (reproducible results)
rng = random.Random(42)

print("Loading phishing dataset...")
phishing_df = pd.read_csv("data/phishing_site_urls.csv")

# The raw file has columns "URL" and "Label" ("bad" / "good")
phishing_df = phishing_df.rename(columns={"URL": "url", "Label": "label"})

# Phishing URLs: only the "bad" rows
phishing_only = phishing_df[phishing_df["label"] == "bad"][["url"]].copy()
phishing_only["label"] = 1   # 1 = phishing

# Raw URLs have no http/https prefix; add one for consistency
phishing_only["url"] = phishing_only["url"].apply(
    lambda u: u if u.startswith(("http://", "https://")) else "http://" + u
)

print(f"Phishing URLs found: {len(phishing_only)}")

print("Loading Tranco top sites (legitimate domains)...")
# The Tranco list has no header, format: rank,domain
tranco_df = pd.read_csv("data/top-1m.csv", header=None, names=["rank", "domain"])

# Random sample of the whole 1M list (not just the top-ranked famous sites)
tranco_df = tranco_df.sample(n=50000, random_state=42)


# ---- Make the synthetic safe URLs look like real websites ----
# Real sites use www., sub-domains and paths of every length, often with digits and hyphens
# (for example /location/330-city/digital-menu). If the safe examples are short and the
# phishing examples are long, the model learns "long URL = phishing", and legitimate long
# links (restaurant menus, shop pages, news articles) are wrongly flagged.
PLAIN_PATHS = [
    "/", "/about", "/contact", "/products", "/products/12345",
    "/blog/how-to-get-started", "/careers", "/help/faq", "/docs/getting-started",
    "/en/pricing", "/news/2024/05/update", "/search?q=shoes", "/store/item/48213",
]
# A few real safe URLs contain "login" / "account"; keep a small share so these words
# alone do not become a shortcut for phishing
KEYWORD_PATHS = ["/login", "/signin?next=/dashboard", "/account/settings", "/secure/checkout"]
OTHER_SUBDOMAINS = ["blog.", "docs.", "support.", "api.", "shop."]

# Building blocks for long, realistic paths
SECTIONS = ["location", "store", "product", "category", "article", "menu", "shop",
            "events", "support/articles", "restaurants", "collections"]
CITIES = ["lincoln-ne", "nagpur", "pune-central", "mumbai-west", "bengaluru", "chennai-north",
          "delhi-ncr", "hyderabad", "kolkata", "ahmedabad"]
SLUGS = ["digital-menu", "getting-started", "pricing-and-plans", "how-to-track-your-order",
         "summer-collection-2024", "customer-reviews", "returns-and-refunds",
         "annual-report", "contact-us", "store-locator", "weekly-offers", "new-arrivals"]


def make_long_path() -> str:
    """A long, realistic path with digits and hyphens, like /location/330-city/digital-menu."""
    section = rng.choice(SECTIONS)
    r = rng.random()
    if r < 0.40:
        return f"/{section}/{rng.randint(10, 99999)}-{rng.choice(CITIES)}/{rng.choice(SLUGS)}"
    if r < 0.70:
        return f"/{section}/{rng.choice(SLUGS)}/{rng.randint(1000, 999999)}"
    return f"/{rng.randint(2015, 2025)}/{rng.randint(1, 12):02d}/{rng.choice(SLUGS)}-{rng.choice(SLUGS)}"


def make_safe_url(domain: str) -> str:
    r = rng.random()
    if r < 0.55:
        sub = ""                               # 55%: bare domain
    elif r < 0.85:
        sub = "www."                           # 30%: www.
    else:
        sub = rng.choice(OTHER_SUBDOMAINS)     # 15%: blog., docs. ...

    r = rng.random()
    if r < 0.25:
        path = ""                              # 25%: no path
    elif r < 0.60:
        path = rng.choice(PLAIN_PATHS)         # 35%: normal short path
    elif r < 0.66:
        path = rng.choice(KEYWORD_PATHS)       # 6%: login/account style path
    else:
        path = make_long_path()                # 34%: long realistic path
    return "https://" + sub + domain + path


# ---- SAFE URLs: real "good" URLs (Kaggle) + modern domains with synthetic paths (Tranco) ----
N_REAL_SAFE = 1400
N_SYNTH_SAFE = 1600

good_df = phishing_df[phishing_df["label"] == "good"][["url"]].copy()
good_df["url"] = good_df["url"].apply(
    lambda u: u if u.startswith(("http://", "https://")) else "http://" + u
)
good_df["host"] = (good_df["url"].str.replace(r"^https?://", "", regex=True)
                   .str.split("/").str[0].str.lower())
good_df = good_df[good_df["host"].str.contains(".", regex=False)]   # drop junk rows
# At most 2 URLs per domain, so one or two big sites cannot fill the whole dataset
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


# ---- Length-matched sampling ----
# In the raw data phishing URLs are much longer than safe ones, so URL length alone would
# almost decide the label. For every length range we keep the same number of safe and
# phishing URLs, so length stops being a shortcut and the model must look at other signs.
def match_lengths(a: pd.DataFrame, b: pd.DataFrame, n_bins: int = 20, seed: int = 42):
    """Returns subsets of a and b that have the same number of URLs in every length range."""
    edges = np.unique(np.quantile(a["url"].str.len(), np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf     # catch very short and very long URLs too
    bin_a = pd.cut(a["url"].str.len(), bins=edges)
    bin_b = pd.cut(b["url"].str.len(), bins=edges)
    parts_a, parts_b = [], []
    for interval in bin_a.cat.categories:
        group_a = a[bin_a == interval]
        group_b = b[bin_b == interval]
        k = min(len(group_a), len(group_b))
        if k == 0:
            continue
        parts_a.append(group_a.sample(n=k, random_state=seed))
        parts_b.append(group_b.sample(n=k, random_state=seed))
    return pd.concat(parts_a, ignore_index=True), pd.concat(parts_b, ignore_index=True)


def describe(name: str, df: pd.DataFrame) -> None:
    lengths = df["url"].str.len()
    has_path = df["url"].str.contains(r"://[^/]+/.+", regex=True).mean()
    print(f"   {name}: {len(df)} URLs, mean length {lengths.mean():.1f}, "
          f"median {lengths.median():.0f}, with a path {has_path:.0%}")


print("\nBefore length matching:")
describe("safe    ", safe_only)
describe("phishing", phishing_only)

safe_sample, phishing_sample = match_lengths(safe_only, phishing_only)

print("After length matching:")
describe("safe    ", safe_sample)
describe("phishing", phishing_sample)

# ---- HARD EXAMPLES: make the dataset more like the real world ----

# Hard NEGATIVES: legitimate URLs that look like phishing (long, complex)
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

# Hard POSITIVES: phishing URLs that look like safe ones (short, clean)
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

# ---- Combine and shuffle ----
final_df = pd.concat(
    [phishing_sample, safe_sample, hard_safe_urls, hard_phishing_urls],
    ignore_index=True
)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ---- Save the final file ----
final_df.to_csv("data/phishing_urls.csv", index=False)

print("\nFinal dataset saved to data/phishing_urls.csv")
print(f"   Total: {len(final_df)} URLs")
print(f"   Phishing: {(final_df['label'] == 1).sum()}")
print(f"   Safe: {(final_df['label'] == 0).sum()}")
