import re                                  # regex (pattern matching) ke liye, IP address check karne mein use hoga
from urllib.parse import urlparse           # URL ko parts mein todne ke liye
import tldextract                           # domain, subdomain, suffix (.com/.in) alag karne ke liye library

# Offline suffix list use karo (internet se live fetch mat karo -- fast aur reliable rehta hai)
_tld_extractor = tldextract.TLDExtract(suffix_list_urls=())

# Ye words jo phishing URLs mein commonly milte hain
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "account", "update", "secure", "banking",
    "confirm", "signin", "webscr", "password", "billing", "suspend"
]

# Ye famous brand names hain -- agar ye kisi fake subdomain mein milein to red flag hai
BRAND_NAMES = [
    "paypal", "google", "microsoft", "apple", "amazon", "netflix",
    "facebook", "instagram", "whatsapp", "sbi", "hdfc", "icici"
]


def extract_url_features(url: str) -> dict:
    parsed = urlparse(url if "://" in url else "http://" + url)
    ext = _tld_extractor(url)
    hostname = parsed.hostname or ""
    domain = ext.domain.lower()
    full_url = url.lower()

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "has_at_symbol": int("@" in url),
        "has_ip_address": int(bool(re.match(
            r"^(\d{1,3}\.){3}\d{1,3}$", hostname))),
        #"uses_https": int(parsed.scheme == "https"),
        "num_subdomains": max(ext.subdomain.count(".") + 1, 0)
            if ext.subdomain else 0,
        "has_suspicious_keyword": int(
            any(kw in full_url for kw in SUSPICIOUS_KEYWORDS)),
        "brand_in_subdomain_not_domain": int(
            any(b in (ext.subdomain or "").lower() and b != domain
                for b in BRAND_NAMES)),
        "num_redirect_chars": url.count("//") - 1,
        "is_shortened": int(any(s in hostname for s in [
            "bit.ly", "tinyurl", "t.co", "goo.gl", "is.gd"])),
    }
    return features