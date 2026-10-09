import os
import tempfile
import traceback
from datetime import datetime

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Only agents whose dependencies are always installed are imported at the top level.
# qr (opencv), account (training data) and apk (androguard) are imported lazily inside
# their endpoints, so a problem with one agent can't bring the whole API down at startup.
from agents import url_agent, email_agent, xai_agent, report_agent

app = FastAPI(title="CyberLens API", version="3.0")

# allow_origins=["*"] means requests are accepted from anywhere (fine for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# These verdicts are treated as "dangerous" -- risk_score is calculated from this
RISKY_VERDICTS = {"phishing", "malicious", "suspicious"}

MAX_EMAIL_CHARS = 20000              # very long text shouldn't slow the server down
MAX_QR_BYTES = 5 * 1024 * 1024       # QR image max 5 MB
MAX_APK_BYTES = 25 * 1024 * 1024     # APK max 25 MB (keeps the free-tier memory in mind)

# Account-takeover demo: the agent's own out-of-sample demo logs, with a fixed "end_time"
ATO_DEMO_END = "2026-10-01T00:00:00"
ATO_SCENARIOS = {
    "quiet_normal": "Ordinary month of chatting with friends",
    "party_invite": "One invite with a link sent to 18 friends within minutes",
    "friend_money_scam": "Urgent money request sent to every friend at 3 AM",
    "slow_low_attacker": "Slow trickle of personalised delivery-failed links to strangers",
}


class URLRequest(BaseModel):
    url: str


class EmailRequest(BaseModel):
    text: str


class AccountRequest(BaseModel):
    scenario: str


def _pct(confidence: float) -> int:
    """Turns confidence into a whole-number percent, exactly like the report text
    (so the app banner and the report never disagree, e.g. 97% vs 96%)."""
    return int(f"{confidence:.0%}"[:-1])


def build_response(agent: str, kind: str, result: dict, extra: dict = None) -> dict:
    """
    Turns any ML agent's result (containing 'model' and 'feature_row') into a common
    JSON format with a SHAP explanation and a report.
    This lets the Flutter app use a single result screen for every agent.
    """
    xai_result = xai_agent.explain(result)
    report_text = report_agent.generate_report(result, xai_result, kind=kind)

    confidence = result["confidence"]
    risky = result["verdict"] in RISKY_VERDICTS
    risk_score = round(confidence * 100) if risky else round((1 - confidence) * 100)

    response = {
        "agent": agent,
        "verdict": result["verdict"],
        "confidence": confidence,
        "confidence_pct": _pct(confidence),
        "risk_score": risk_score,               # 0 = completely safe, 100 = very risky
        "top_features": xai_result["top_features"],
        "explanation_type": "shap",
        "report": report_text,
    }
    if extra:
        response.update(extra)
    return response


def _read_upload(file: UploadFile, max_bytes: int, what: str) -> bytes:
    """Reads an upload into memory, with a size limit."""
    data = file.file.read(max_bytes + 1)
    if not data:
        raise HTTPException(status_code=400, detail=f"{what} is empty.")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{what} is too large (max {max_bytes // (1024 * 1024)} MB).",
        )
    return data


@app.get("/")
def health_check():
    """Simple endpoint to check that the API is alive."""
    return {
        "status": "CyberLens API is running",
        "endpoints": [
            "/analyze/url", "/analyze/email", "/analyze/qr",
            "/analyze/account", "/analyze/apk",
        ],
    }


# ---------------------------------------------------------------- URL
@app.post("/analyze/url")
def analyze_url(request: URLRequest):
    """URL -> ML prediction -> SHAP explanation -> Report"""
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is empty.")
    result = url_agent.predict(url)
    return build_response("url", "url", result,
                          {"url": result["url"], "features": result["features"]})


# -------------------------------------------------------------- EMAIL
@app.post("/analyze/email")
def analyze_email(request: EmailRequest):
    """Email text -> TF-IDF + heuristics -> SHAP explanation -> Report"""
    text = request.text.strip()[:MAX_EMAIL_CHARS]
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Email text is too short. Paste the full email.")
    result = email_agent.predict(text)
    # The 50 tfidf word-columns aren't useful for the UI, so we only send the heuristic features
    heuristics = {k: v for k, v in result["features"].items() if not k.startswith("tfidf_")}
    return build_response("email", "email", result, {"features": heuristics})


# ----------------------------------------------------------------- QR
@app.post("/analyze/qr")
def analyze_qr(file: UploadFile = File(...)):
    """QR image -> decode (OpenCV) -> URL agent -> SHAP explanation -> Report"""
    data = _read_upload(file, MAX_QR_BYTES, "Image")
    try:
        from agents import qr_agent          # cv2 is loaded here
    except ImportError:
        raise HTTPException(status_code=503, detail="QR module is not installed on the server (opencv).")

    # qr_agent takes a file PATH, not bytes -- so we write a temp file
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload.png")   # OpenCV looks at the content, not the extension
        with open(path, "wb") as f:
            f.write(data)
        try:
            result = qr_agent.predict(path)
        except ValueError:
            raise HTTPException(status_code=400, detail="Could not read the image. Please send a PNG or JPG.")

    if result["verdict"] == "unknown":
        # In this case qr_agent returns model=None, so SHAP can't run
        raise HTTPException(status_code=422, detail="No QR code found in this image. Take a clear, straight-on photo.")

    # The QR result is really a URL result, so the report kind is "url"
    return build_response("qr", "url", result, {
        "url": result["url"],
        "decoded_url": result["decoded_url"],
        "features": result["features"],
    })


# ------------------------------------------------------------ ACCOUNT
def _ensure_ato_model(ato):
    """Unlike the url/email agents, this agent doesn't train itself -- if the model is missing, we train it here."""
    if ato.MODEL_PATH.exists():
        return
    try:
        ato.train()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Account-takeover model or training data is not available on the server.",
        )


@app.get("/analyze/account/scenarios")
def account_scenarios():
    """List of demo scenarios for the Flutter dropdown."""
    return {"scenarios": [{"id": k, "description": v} for k, v in ATO_SCENARIOS.items()]}


@app.post("/analyze/account")
def analyze_account(request: AccountRequest):
    """Simulated message logs for a demo scenario -> graph features -> SHAP -> Report"""
    scenario = request.scenario.strip()
    if scenario not in ATO_SCENARIOS:
        raise HTTPException(status_code=400, detail=f"Unknown scenario. Options: {list(ATO_SCENARIOS)}")

    from agents import account_takeover_agent as ato
    _ensure_ato_model(ato)

    logs = ato._build_demo_logs(scenario)
    result = ato.predict({"account_id": "demo_user", "message_logs": logs, "end_time": ATO_DEMO_END})
    return build_response("account_takeover", "account", result, {
        "account_id": "demo_user",
        "scenario": scenario,
        "scenario_description": ATO_SCENARIOS[scenario],
        "features": result["features"],
    })


# ---------------------------------------------------------------- APK
def _apk_report_text(result: dict, risk_pct: int, top_features: list) -> str:
    actions = {
        "malicious": "Do NOT install this APK. If it is already installed, uninstall it and report it to your security team.",
        "suspicious": "Install only if you fully trust the source. Legitimate apps (e.g. messaging apps) can also request these permissions, so prefer the official Play Store version.",
        "safe": "No strong indicators were found by the permission checks. This is not a guarantee - install apps only from trusted sources.",
    }
    factors = "\n".join(
        f"- {name}: {'increased' if weight > 0 else 'decreased'} risk ({weight:+.3f})"
        for name, weight in top_features
    ) or "- No sensitive permission patterns found"
    evidence = "\n".join(f"- {e}" for e in result["evidence"])
    method = (
        "RandomForest malware probability, explained with SHAP"
        if result["explanation_type"] == "shap"
        else "rule-based heuristic (SHAP unavailable)"
    )
    caveat = ""
    if result["model_source"].startswith("synthetic"):
        caveat = (
            "\nNote: the model was trained on synthetic permission profiles, "
            "not real malware samples, so treat this score as indicative only.\n"
        )
    return (
        "CyberLens APK Report\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"App: {result.get('app_name') or 'unknown'}\n"
        f"Package: {result.get('package_name') or 'unknown'}\n"
        f"SHA-256: {result['sha256']}\n"
        f"Verdict: {result['verdict'].upper()}\n"
        f"Risk score: {risk_pct}/100 ({method})\n"
        f"Model training data: {result['model_source']}\n"
        f"{caveat}\n"
        f"Key Contributing Factors (permissions requested by this APK):\n{factors}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Recommended Action:\n{actions[result['verdict']]}"
    ).strip()


@app.post("/analyze/apk")
def analyze_apk_endpoint(file: UploadFile = File(...)):
    """APK upload -> Androguard permission analysis -> RandomForest + SHAP -> Report."""
    if not (file.filename or "").lower().endswith(".apk"):
        raise HTTPException(status_code=400, detail="Please upload an .apk file only.")
    data = _read_upload(file, MAX_APK_BYTES, "APK")
    try:
        from agents import apk_agent         # androguard is loaded here
    except ImportError:
        raise HTTPException(status_code=503, detail="APK module is not installed on the server (androguard).")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload.apk")
        with open(path, "wb") as f:
            f.write(data)
        try:
            result = apk_agent.analyze_apk(path)
        except Exception as exc:
            traceback.print_exc()        # the real traceback shows up in Render's Logs
            raise HTTPException(
                status_code=422,
                detail=f"This APK could not be read ({type(exc).__name__}: {str(exc)[:200]})",
            )

    risk_pct = round(result["risk_score"] * 100)
    top_features = result["top_features"]

    return {
        "agent": "apk",
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "confidence_pct": _pct(result["confidence"]),
        "risk_score": risk_pct,
        "top_features": top_features,
        "explanation_type": result["explanation_type"],   # "shap" (or "heuristic" if SHAP failed)
        "model_source": result["model_source"],
        "package_name": result.get("package_name"),
        "app_name": result.get("app_name"),
        "sha256": result["sha256"],
        "permission_count": len(result["permissions"]),
        "risky_permissions": result["risky_permissions"],
        "evidence": result["evidence"],
        "report": _apk_report_text(result, risk_pct, top_features),
    }