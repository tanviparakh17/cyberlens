from datetime import datetime   # current time/date nikaalne ke liye


def generate_report(url_result: dict, xai_result: dict) -> str:
    verdict = url_result["verdict"]
    confidence = url_result["confidence"]
    top_features = xai_result["top_features"]

    reasons = "\n".join(
        f"- {name}: {'increased' if val > 0 else 'decreased'} phishing risk (impact {val:.3f})"
        for name, val in top_features
    )

    action = (
        "Do NOT click this link or enter any credentials. Report it to your security team."
        if verdict == "phishing"
        else "This URL appears safe based on current analysis, but always verify the sender."
    )

    report = f"""CyberLens Threat Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

URL Analyzed: {url_result['url']}
Verdict: {verdict.upper()}
Confidence: {confidence:.0%}

Key Contributing Factors:
{reasons}

Recommended Action:
{action}
"""
    return report.strip()