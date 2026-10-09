import sys
sys.path.insert(0, ".")
from agents import url_agent

SAFE = [
    "https://www.bbc.com/news/world-asia-india-12345678",
    "https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster",
    "https://www.coursera.org/learn/machine-learning",
    "https://kiranagrocery.in/shop/rice-5kg",
    "https://www.sbi.co.in/web/personal-banking/accounts",
    "https://netbanking.hdfcbank.com/netbanking/",
    "https://support.apple.com/en-in/HT201994",
    "https://www.irctc.co.in/nget/train-search",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://www.techblog-weekly.com/2024/10/best-laptops-under-50000.html",
    "https://sunrise-yoga-studio.com/classes/beginner",
    "https://my-portfolio.dev/projects/cyberlens",
]
PHISHING = [
    "http://sbi-onlinebanking-verify.com/login.php",
    "http://secure.paypaI-support.com/webscr?cmd=_login",
    "http://amazon-prize-winner.top/claim?id=77231",
    "http://netflix.billing-update.xyz/account/payment",
    "http://login-microsoft365.verify-session.co/auth",
    "http://hdfc-netbanking-update.in/kyc/verify",
    "http://faceb00k.security-check.ml/login",
    "http://dhl-parcel-tracking.cf/pay-customs?ref=88213",
    "http://instagram-verify-badge.site/apply",
    "http://icici-rewards-points.tk/redeem",
    "http://www.g00gle-docs-share.com/view?doc=5512",
    "http://bank-alert.support-center.link/verify/otp",
]

def run(urls, expected):
    ok = 0
    for u in urls:
        r = url_agent.predict(u)
        v = r["verdict"]
        if v == expected:
            ok += 1
        else:
            print(f"  MISS ({v}, {r.get('source')}, {r.get('reason')}): {u}")
    return ok

print("Safe URLs:")
s = run(SAFE, "safe")
print("Phishing URLs:")
p = run(PHISHING, "phishing")
print(f"\nSafe correct: {s}/{len(SAFE)}   Phishing caught: {p}/{len(PHISHING)}")