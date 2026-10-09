import sys
sys.path.insert(0, ".")
from agents import url_agent

# (URL, expected verdict)
CHECKS = [
    ("https://www.wikipedia.org", "safe"),
    ("https://docs.python.org/3/tutorial/", "safe"),
    ("https://blog.google/products/", "safe"),
    ("http://api.demo.dev/contact", "safe"),
    ("https://rbunagpur.in/", "safe"),
    ("https://www.iitb.ac.in", "safe"),
    ("https://www.flipkart.com/mobiles", "safe"),
    ("https://github.com/anthropics", "safe"),
    ("https://www.amazon.in/gp/cart", "safe"),
    ("http://paypal-account-verify.suspend-secure-login.xyz", "phishing"),
    ("http://paypal-verify-login.tk", "phishing"),
    ("http://amaz0n-verify.com", "phishing"),
    ("http://appIe-icloud-verify.com", "phishing"),
    ("http://secure-hdfc.in/login/update?id=8842", "phishing"),
    ("http://192.168.4.7/bank/signin.php", "phishing"),
    ("http://wellsfarg0-alert.net/verify/account.php", "phishing"),
    ("https://stackoverflow.com/questions/12345/how-to-parse-json", "safe"),
("https://www.hdfcbank.com/personal/pay/cards", "safe"),
("https://accounts.google.com/signin/v2/identifier", "safe"),
("http://sbi-kyc-update.top/login", "phishing"),
("http://micros0ft-support-alert.com/verify", "phishing"),
]

ok = 0
for url, expected in CHECKS:
    r = url_agent.predict(url)
    good = r["verdict"] == expected
    ok += good
    print("OK  " if good else "FAIL", f"{r['verdict']:8s} {r['confidence']:.2f}  {url}")
print(f"=> {ok}/{len(CHECKS)} correct")