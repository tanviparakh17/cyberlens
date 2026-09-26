from fastapi import FastAPI                      # web API framework
from fastapi.middleware.cors import CORSMiddleware  # browser/app se requests allow karne ke liye
from pydantic import BaseModel                    # request data validate karne ke liye

from agents import url_agent, xai_agent, report_agent

app = FastAPI(title="CyberLens API", version="1.0")

# CORS zaroori hai taaki mobile app / browser extension is API ko call kar sake
# allow_origins=["*"] matlab kisi bhi jagah se request accept hogi (development ke liye theek hai)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request body ka structure define kar rahe hain -- client ko sirf "url" bhejna hoga
class URLRequest(BaseModel):
    url: str


@app.get("/")
def health_check():
    """Simple endpoint check karne ke liye ki API zinda hai ya nahi."""
    return {"status": "CyberLens API is running"}


@app.post("/analyze/url")
def analyze_url(request: URLRequest):
    """
    Poora pipeline: URL -> ML prediction -> SHAP explanation -> Report
    Mobile app / frontend isko POST request se call karega, JSON body mein {"url": "..."}
    """
    url_result = url_agent.predict(request.url)
    xai_result = xai_agent.explain(url_result)
    report_text = report_agent.generate_report(url_result, xai_result)

    # 'model' aur 'feature_row' ko response mein nahi bhej sakte (JSON serializable nahi hain)
    # isliye sirf zaroori cheezein return kar rahe hain
    return {
        "url": url_result["url"],
        "verdict": url_result["verdict"],
        "confidence": url_result["confidence"],
        "features": url_result["features"],
        "top_features": xai_result["top_features"],
        "report": report_text,
    }