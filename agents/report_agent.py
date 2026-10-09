from datetime import datetime   # current time/date nikaalne ke liye

# Jin verdicts ko "khatre wala" maana jaata hai (koi bhi agent ho)
RISKY_VERDICTS = {"phishing", "malicious", "suspicious"}

# Har agent type ke liye: input ka label, aur risky / safe hone par recommended action
KINDS = {
    "url": {
        "label": "URL Analyzed",
        "risky": "Do NOT click this link or enter any credentials. Report it to your security team.",
        "safe": "This URL appears safe based on current analysis, but always verify the sender.",
    },
    "email": {
        "label": "Email Analyzed",
        "risky": "Do NOT click any link or open any attachment. Do not reply. Report it as phishing and delete it.",
        "safe": "This email looks legitimate based on current analysis, but verify the sender before sharing sensitive information.",
    },
    "qr": {
        "label": "QR Code Points To",
        "risky": "Do NOT open the link behind this QR code or enter any details on it.",
        "safe": "The link behind this QR code looks safe, but be careful with QR stickers pasted over other QR codes.",
    },
    "account": {
        "label": "Account Analyzed",
        "risky": "Change the password now, log out of all devices, turn on 2-step verification, and warn contacts not to click links sent from this account.",
        "safe": "No takeover pattern found in this account's recent activity. Keep 2-step verification on.",
    },
}


def _subject(result: dict, kind: str) -> str:
    """Report mein dikhane ke liye input ka chhota sa text nikaalta hai."""
    if kind == "email":
        text = " ".join(str(result.get("email_text", "")).split())   # extra spaces/newlines hatao
        return text[:150] + ("..." if len(text) > 150 else "")
    if kind == "qr":
        return result.get("decoded_url") or "(no link found)"
    if kind == "account":
        return str(result.get("account_id", "unknown"))
    return str(result.get("url", ""))


def _nice(name: str) -> str:
    """Feature ka naam readable banata hai (email ke tfidf_ features ke liye bhi)."""
    if name.startswith("tfidf_"):
        return f'word "{name[6:]}"'
    return name.replace("_", " ")


def generate_report(result: dict, xai_result: dict, kind: str = "url") -> str:
    info = KINDS.get(kind, KINDS["url"])
    verdict = result["verdict"]
    confidence = result["confidence"]

    reasons = "\n".join(
        f"- {_nice(name)}: {'increased' if val > 0 else 'decreased'} risk (impact {val:.3f})"
        for name, val in xai_result["top_features"]
    )

    action = info["risky"] if verdict in RISKY_VERDICTS else info["safe"]

    report = f"""CyberLens Threat Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

{info['label']}: {_subject(result, kind)}
Verdict: {verdict.upper()}
Confidence: {confidence:.0%}

Key Contributing Factors:
{reasons}

Recommended Action:
{action}
"""
    return report.strip()