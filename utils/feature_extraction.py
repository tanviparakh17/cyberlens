import re
from urllib.parse import urlparse
import tldextract

_tld_extractor = tldextract.TLDExtract(suffix_list_urls=())

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "account", "update", "secure", "banking",
    "confirm", "signin", "webscr", "password", "billing", "suspend"
]

BRAND_NAMES = [
    "paypal", "google", "microsoft", "apple", "amazon", "netflix",
    "facebook", "instagram", "whatsapp", "sbi", "hdfc", "icici",
    "icloud", "wellsfargo", "paytm", "flipkart", "linkedin", "axis"
]

# Brands ke asli registered domains. Inpe impersonation flag nahi lagega.
OFFICIAL_DOMAINS = {
    "paypal.com", "google.com", "microsoft.com", "apple.com", "icloud.com",
    "amazon.com", "amazon.in", "netflix.com", "facebook.com", "instagram.com",
    "whatsapp.com", "sbi.co.in", "hdfcbank.com", "hdfc.com", "icicibank.com",
    "wellsfargo.com", "paytm.com", "flipkart.com", "linkedin.com",
    "axisbank.com", "amazonaws.com", "googleapis.com", "googleusercontent.com",
    "microsoftonline.com",
}

# Phishing mein aam use hone wale cheap/free TLDs
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "link", "work",
    "icu", "cyou", "buzz", "rest", "monster", "zip", "support",
}

# Homoglyph map: 0->o, 3->e, 4->a, 5->s, @->a, $->s, 1->l
_LEET = str.maketrans({"0": "o", "3": "e", "4": "a", "5": "s",
                       "@": "a", "$": "s", "1": "l"})


def _raw_hostname(url: str) -> str:
    """Case preserve karke hostname deta hai (urlparse().hostname lowercase kar deta hai)."""
    netloc = urlparse(url if "://" in url else "http://" + url).netloc
    netloc = netloc.rsplit("@", 1)[-1]
    return netloc.split(":")[0]


def _normalize(host: str) -> str:
    """Lowercase se PEHLE capital I -> l, phir leet chars aur rn->m."""
    host = host.replace("I", "l").lower()
    host = host.replace("rn", "m")
    return host.translate(_LEET)


def _find_brand(text: str) -> str:
    tokens = re.split(r"[-.]", text)
    for b in BRAND_NAMES:
        if len(b) <= 4:          # sbi/hdfc/axis: sirf poore token pe match (false positives se bachne ke liye)
            if b in tokens:
                return b
        elif b in text:
            return b
    return ""


def extract_url_features(url: str) -> dict:
    parsed = urlparse(url if "://" in url else "http://" + url)
    ext = _tld_extractor(url)
    hostname = parsed.hostname or ""
    domain = ext.domain.lower()
    full_url = url.lower()

    has_ip = bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$", hostname))

    # ---- Patch 2b: hostname-level features ----
    raw_host = _raw_hostname(url)
    plain_host = raw_host.lower()
    norm_host = _normalize(raw_host)

    ext_plain = _tld_extractor(plain_host)
    ext_norm = _tld_extractor(norm_host)

    # Brand match sirf subdomain+domain mein, suffix mein nahi (warna blog.google false flag hoga)
    plain_text = ".".join(p for p in (ext_plain.subdomain, ext_plain.domain) if p)
    norm_text = ".".join(p for p in (ext_norm.subdomain, ext_norm.domain) if p)

    plain_brand = _find_brand(plain_text)
    norm_brand = _find_brand(norm_text)
    norm_domain_brand = _find_brand(ext_norm.domain)

    registered_plain = f"{ext_plain.domain}.{ext_plain.suffix}" if ext_plain.suffix else ext_plain.domain
    is_official = registered_plain in OFFICIAL_DOMAINS

    if has_ip:
        brand_impersonation = homoglyph_brand = hyphen_brand_domain = 0
        suspicious_tld = domain_has_leet = domain_hyphens = 0
    else:
        brand_impersonation = int(bool(norm_brand) and not is_official)
        homoglyph_brand = int(bool(norm_brand) and not plain_brand and not is_official)
        hyphen_brand_domain = int(bool(norm_domain_brand)
                                  and "-" in ext_plain.domain and not is_official)
        suspicious_tld = int(ext_plain.suffix.split(".")[-1] in SUSPICIOUS_TLDS)
        domain_has_leet = int(bool(re.search(r"[a-z][0-9][a-z]", ext_plain.domain)))
        domain_hyphens = ext_plain.domain.count("-")

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "has_at_symbol": int("@" in url),
        "has_ip_address": int(has_ip),
        # "uses_https": int(parsed.scheme == "https"),
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
        # naye features
        "brand_impersonation": brand_impersonation,
        "homoglyph_brand": homoglyph_brand,
        "hyphen_brand_domain": hyphen_brand_domain,
        "suspicious_tld": suspicious_tld,
        "domain_has_leet": domain_has_leet,
        "domain_hyphens": domain_hyphens,
    }
    return features

def get_registered_domain(url: str) -> str:
    """example: https://docs.python.org/3/ -> python.org"""
    ext = _tld_extractor(_raw_hostname(url).lower())
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain