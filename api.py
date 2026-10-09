from fastapi import FastAPI, HTTPException            # web API framework
from fastapi.middleware.cors import CORSMiddleware    # mobile app se requests allow karne ke liye
from pydantic import BaseModel                        # request data validate karne ke liye

from agents import url_agent, email_agent, xai_agent, report_agent

app = FastAPI(title="CyberLens API", version="2.0")

# allow_origins=["*"] matlab kisi bhi jagah se request accept hogi (development ke liye theek hai)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ye verdicts "khatre wale" maane jaate hain -- risk_score isi se calculate hota hai
RISKY_VERDICTS = {"phishing", "malicious", "suspicious"}
MAX_EMAIL_CHARS = 20000   # bahut lamba text aaye to server dheema na ho


class URLRequest(BaseModel):
    url: str


class EmailRequest(BaseModel):
    text: str


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
        "report": report_text,
    }
    if extra:
        response.update(extra)
    return response


@app.get("/")
def health_check():
    """Simple endpoint check karne ke liye ki API zinda hai ya nahi."""
    return {"status": "CyberLens API is running", "endpoints": ["/analyze/url", "/analyze/email"]}


@app.post("/analyze/url")
def analyze_url(request: URLRequest):
    """URL -> ML prediction -> SHAP explanation -> Report"""
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL khaali hai.")
    result = url_agent.predict(url)
    return build_response("url", "url", result,
                          {"url": result["url"], "features": result["features"]})


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