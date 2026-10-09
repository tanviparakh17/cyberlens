import os
import tempfile
from datetime import datetime

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Sirf wahi agents top-level pe import hote hain jinki dependencies hamesha installed hain.
# qr (opencv), account (training data) aur apk (androguard) endpoint ke andar lazily import
# hote hain, taaki kisi ek agent ki dikkat poori API ko startup pe hi na gira de.
from agents import url_agent, email_agent, xai_agent, report_agent

app = FastAPI(title="CyberLens API", version="3.0")

# allow_origins=["*"] matlab kisi bhi jagah se request accept hogi (development ke liye theek hai)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ye verdicts "khatre wale" maane jaate hain -- risk_score isi se calculate hota hai
RISKY_VERDICTS = {"phishing", "malicious", "suspicious"}

MAX_EMAIL_CHARS = 20000              # bahut lamba text aaye to server dheema na ho
MAX_QR_BYTES = 5 * 1024 * 1024       # QR image max 5 MB
MAX_APK_BYTES = 25 * 1024 * 1024     # APK max 25 MB (free-tier memory ko dhyan mein rakh ke)

# Account-takeover demo: agent ke apne out-of-sample demo logs, ek fixed "end_time" ke saath
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
    """Confidence ko whole-number percent banata hai, bilkul report ke text jaisa
    (taaki app ke banner aur report mein 97% vs 96% ka farak na aaye)."""
    return int(f"{confidence:.0%}"[:-1])


def build_response(agent: str, kind: str, result: dict, extra: dict = None) -> dict:
    """
    Kisi bhi ML agent ke result (jisme 'model' aur 'feature_row' ho) ko
    SHAP explanation + report ke saath ek common JSON format mein badalta hai.
    Isse Flutter app mein har agent ke liye ek hi result screen chal sakti hai.
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
        "risk_score": risk_score,               # 0 = bilkul safe, 100 = bahut risky
        "top_features": xai_result["top_features"],
        "explanation_type": "shap",
        "report": report_text,
    }
    if extra:
        response.update(extra)
    return response


def _read_upload(file: UploadFile, max_bytes: int, what: str) -> bytes:
    """Upload ko memory mein padhta hai, size limit ke saath."""
    data = file.file.read(max_bytes + 1)
    if not data:
        raise HTTPException(status_code=400, detail=f"{what} khaali hai.")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{what} bahut badi hai (max {max_bytes // (1024 * 1024)} MB).",
        )
    return data


@app.get("/")
def health_check():
    """Simple endpoint check karne ke liye ki API zinda hai ya nahi."""
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
        raise HTTPException(status_code=400, detail="URL khaali hai.")
    result = url_agent.predict(url)
    return build_response("url", "url", result,
                          {"url": result["url"], "features": result["features"]})


# -------------------------------------------------------------- EMAIL
@app.post("/analyze/email")
def analyze_email(request: EmailRequest):
    """Email text -> TF-IDF + heuristics -> SHAP explanation -> Report"""
    text = request.text.strip()[:MAX_EMAIL_CHARS]
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Email text bahut chhota hai. Poora email paste karo.")
    result = email_agent.predict(text)
    # 50 tfidf word-columns UI ke kaam ke nahi, sirf heuristic features bhejte hain
    heuristics = {k: v for k, v in result["features"].items() if not k.startswith("tfidf_")}
    return build_response("email", "email", result, {"features": heuristics})


# ----------------------------------------------------------------- QR
@app.post("/analyze/qr")
def analyze_qr(file: UploadFile = File(...)):
    """QR image -> decode (OpenCV) -> URL agent -> SHAP explanation -> Report"""
    data = _read_upload(file, MAX_QR_BYTES, "Image")
    try:
        from agents import qr_agent          # cv2 yahin load hoga
    except ImportError:
        raise HTTPException(status_code=503, detail="QR module server par installed nahi hai (opencv).")

    # qr_agent file PATH leta hai, bytes nahi -- isliye temp file banate hain
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload.png")   # OpenCV extension nahi, content dekhta hai
        with open(path, "wb") as f:
            f.write(data)
        try:
            result = qr_agent.predict(path)
        except ValueError:
            raise HTTPException(status_code=400, detail="Image padhi nahi ja saki. PNG ya JPG bhejo.")

    if result["verdict"] == "unknown":
        # qr_agent is case mein model=None deta hai, SHAP chalega hi nahi
        raise HTTPException(status_code=422, detail="Is image mein QR code nahi mila. Saaf, seedhi photo lo.")

    # QR ka result asal mein URL ka result hai, isliye report ke liye kind="url"
    return build_response("qr", "url", result, {
        "url": result["url"],
        "decoded_url": result["decoded_url"],
        "features": result["features"],
    })


# ------------------------------------------------------------ ACCOUNT
def _ensure_ato_model(ato):
    """url/email agent ki tarah ye agent khud train nahi karta -- model nahi ho to yahin train karte hain."""
    if ato.MODEL_PATH.exists():
        return
    try:
        ato.train()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Account-takeover model ya training data server par nahi hai.",
        )


@app.get("/analyze/account/scenarios")
def account_scenarios():
    """Flutter dropdown ke liye demo scenarios ki list."""
    return {"scenarios": [{"id": k, "description": v} for k, v in ATO_SCENARIOS.items()]}


@app.post("/analyze/account")
def analyze_account(request: AccountRequest):
    """Demo scenario ke simulated message logs -> graph features -> SHAP -> Report"""
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
        f"- {name}: increased risk (weight {weight:.3f})" for name, weight in top_features
    ) or "- No sensitive permission patterns found"
    evidence = "\n".join(f"- {e}" for e in result["evidence"])
    return (
        "CyberLens APK Report\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"App: {result.get('app_name') or 'unknown'}\n"
        f"Package: {result.get('package_name') or 'unknown'}\n"
        f"SHA-256: {result['sha256']}\n"
        f"Verdict: {result['verdict'].upper()}\n"
        f"Risk score: {risk_pct}/100 (rule-based heuristic, not an ML probability)\n\n"
        f"Key Contributing Factors:\n{factors}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Recommended Action:\n{actions[result['verdict']]}"
    ).strip()


@app.post("/analyze/apk")
def analyze_apk_endpoint(file: UploadFile = File(...)):
    """APK upload -> Androguard permission analysis -> rule-based risk score -> Report.
    Is agent mein ML model nahi hai, isliye SHAP nahi -- 'explanation_type' = heuristic."""
    if not (file.filename or "").lower().endswith(".apk"):
        raise HTTPException(status_code=400, detail="Sirf .apk file bhejo.")
    data = _read_upload(file, MAX_APK_BYTES, "APK")
    try:
        from agents import apk_agent         # androguard yahin load hoga
    except ImportError:
        raise HTTPException(status_code=503, detail="APK module server par installed nahi hai (androguard).")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload.apk")
        with open(path, "wb") as f:
            f.write(data)
        try:
            result = apk_agent.analyze_apk(path)
        except Exception:
            raise HTTPException(status_code=422, detail="Ye APK padhi nahi ja saki (corrupt ya unsupported).")

    risk_pct = round(result["risk_score"] * 100)
    contributions = result.get("contributions", [])   # apk_scoring_patch.py lagane ke baad bharta hai
    top_features = [[name, weight] for name, weight in
                    sorted(contributions, key=lambda c: -abs(c[1]))[:5]]

    return {
        "agent": "apk",
        "verdict": result["verdict"],
        "confidence": None,                  # heuristic hai, probability nahi
        "confidence_pct": None,
        "risk_score": risk_pct,
        "top_features": top_features,
        "explanation_type": "heuristic",
        "package_name": result.get("package_name"),
        "app_name": result.get("app_name"),
        "sha256": result["sha256"],
        "permission_count": len(result["permissions"]),
        "risky_permissions": result["risky_permissions"],
        "evidence": result["evidence"],
        "report": _apk_report_text(result, risk_pct, top_features),
    }