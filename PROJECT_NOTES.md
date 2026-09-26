# CyberLens — Project Build Notes

## Day 1 — Backend Rebuild

### Step 1: Folder structure
Created: agents/, utils/, models/, data/
Why: agents/ = specialist AI modules, utils/ = shared helper functions,
     models/ = trained ML files saved here, data/ = dataset CSV goes here

### Step 2: Real dataset sources
- PhishTank (phishtank.org) → phishing URLs ka live feed, free, no login needed
- Tranco (tranco-list.eu) → world's top legitimate websites list, research-grade ranking
- Why two sources: PhishTank sirf bad URLs deta hai, isliye legitimate URLs alag se leni padi

### Step 3: prepare_data.py — DONE
- Kaggle se 156,422 phishing URLs mile (phishing_site_urls.csv)
- Tranco se 1,000,000 legitimate domains mile (top-1m.csv)
- Balanced sample banaya: 2000 phishing + 2000 safe = 4000 total URLs
- Final training file: data/phishing_urls.csv (columns: url, label)

### Step 4: Git/GitHub Setup — DONE
- Repository: github.com/[username]/cyberlens
- .gitignore added: venv, bade dataset CSVs, models excluded
- Team collaborators added via GitHub Settings
- Workflow: git add . -> git commit -m "message" -> git push


### Team Onboarding Steps
1. git clone [repo URL]
2. cd cyberlens, python -m venv venv, activate, pip install -r requirements.txt
3. Dataset files (phishing_site_urls.csv, top-1m.csv) share separately (Google Drive) -- not in Git due to size
4. Future updates: git pull

### Step 6: agents/xai_agent.py — DONE
- SHAP TreeExplainer integrated with RandomForest model
- Returns top 5 contributing features per prediction

### Step 7: Full pipeline tested — DONE
- url_agent -> xai_agent -> report_agent chain verified working

### Step 8: FastAPI backend (api.py) — DONE
- Endpoint: POST /analyze/url
- Input: {"url": "..."}
- Output: verdict, confidence, features, SHAP top_features, full report
- Tested successfully via Swagger UI (localhost:8000/docs) -- 200 OK
- This is what the Flutter mobile app will call
