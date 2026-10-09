import pandas as pd
import tldextract

ext = tldextract.TLDExtract(suffix_list_urls=())

# Free hosting / shorteners: inpe koi bhi phishing page bana sakta hai, allowlist mat karo
EXCLUDE = {
    "blogspot.com", "weebly.com", "wixsite.com", "github.io", "herokuapp.com",
    "netlify.app", "vercel.app", "pages.dev", "web.app", "firebaseapp.com",
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "000webhostapp.com",
    "sites.google.com", "notion.site", "carrd.co", "webflow.io",
}

df = pd.read_csv("data/top-1m.csv", header=None, names=["rank", "domain"])
df = df[pd.to_numeric(df["rank"], errors="coerce").notna()].head(10000)  # header row ho to hata deta hai

trusted = set()
for d in df["domain"].astype(str):
    e = ext(d.strip().lower())
    if e.domain and e.suffix:
        reg = f"{e.domain}.{e.suffix}"
        if reg not in EXCLUDE:
            trusted.add(reg)

with open("data/trusted_domains.txt", "w") as f:
    f.write("\n".join(sorted(trusted)))
print(f"Saved {len(trusted)} trusted domains")