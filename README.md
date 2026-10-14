# CyberLens

CyberLens checks links, emails, QR codes, Android APK files and account activity for signs of
phishing or malware, and explains each verdict in plain terms. This repository is the backend
(a FastAPI service). The mobile app is a separate Flutter project (`cyberlens_app`).

Live API (Render free tier): https://cyberlens-api-ukku.onrender.com  (interactive docs at `/docs`)

> The first request after the server has been idle can take up to about 50 seconds, because the
> free tier sleeps and the models are trained again on the first use.

## What it checks

| Checker | Input | How it decides | Explanation |
|---|---|---|---|
| URL | a link | RandomForest on URL features, plus a rule layer and an allowlist | SHAP |
| Email | pasted email text | TF-IDF words and a few heuristics, RandomForest | SHAP |
| QR | a photo of a QR code | decodes the link with OpenCV, then runs the URL checker | SHAP |
| APK | an Android app file | reads the requested permissions (Androguard), RandomForest | SHAP |
| Account takeover | a simulated scenario | message-graph features (networkx), RandomForest | SHAP |

Verdicts: `safe`, `suspicious`, `phishing` or `malicious` (the account checker uses `normal` and
`suspicious`). Every answer also has a risk score from 0 to 100 and a short text report. Report
times are in IST.

## API

| Method | Path | Body |
|---|---|---|
| POST | `/analyze/url` | JSON `{"url": "..."}` |
| POST | `/analyze/email` | JSON `{"text": "..."}` |
| POST | `/analyze/qr` | multipart form, field `file` (an image) |
| POST | `/analyze/apk` | multipart form, field `file` (an .apk, up to 25 MB) |
| GET | `/analyze/account/scenarios` | list of simulated scenarios |
| POST | `/analyze/account` | JSON `{"scenario": "friend_money_scam"}` |

Response fields: `verdict`, `confidence`, `confidence_pct`, `risk_score`, `top_features`
(a list of `[feature, shap_value]`), `explanation_type`, `report`, and extra fields per checker
(for example `decoded_url` for QR, or `package_name` and `risky_permissions` for APK).

## Project layout

```
api.py                     FastAPI app and endpoints
agents/                    url, email, qr, apk, account_takeover, xai (SHAP), report agents
utils/feature_extraction.py  URL features
data/                      training data, trusted_domains.txt, known_good_domains.txt
models/                    trained models (not in git; created automatically)
prepare_data.py            builds data/phishing_urls.csv
generate_trusted.py        builds data/trusted_domains.txt from the Tranco list
generate_ato_logs.py       builds the simulated account-activity logs
spot_check.py              quick check of 21 example links
```

## Run it locally

```
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

Then open http://localhost:8000/docs. A model that is missing is trained on first use.

To rebuild the URL data you need two large files that are not stored in this repository:
`data/phishing_site_urls.csv` (the Kaggle "Phishing Site URLs" dataset) and `data/top-1m.csv`
(the Tranco top-1M list). Then run `python prepare_data.py` and `python -m agents.url_agent`.

## Results and honest limitations

These numbers are what we measured, not a promise of accuracy in the real world.

- **URL checker.** Domain-separated test accuracy is 0.85 (1798 URLs; no domain appears in both
  training and test data). On 3000 real URLs from the raw dataset whose domains were not used in
  training, a comparable model scored about 0.79. An earlier version showed 0.87, but that came
  from a shortcut: phishing URLs in the data were much longer than safe ones. The training data
  was rebalanced for length and the raw length features were removed. On top of the model there
  are rules (for example brand impersonation, IP address hosts, suspicious endings) and an
  allowlist of well-known domains, so the final verdict is not the model alone. The model cannot
  know the reputation of a new domain, so an unfamiliar legitimate site can still be flagged.
  `spot_check.py` (21 links) was used while tuning the rules, so it is a smoke test, not an
  accuracy measure.
- **APK checker.** There was no labelled APK dataset, so the model is trained on synthetic
  permission profiles. The score is only an indication. Permissions alone do not prove that an
  app is malware. If `data/apk_permissions.csv` exists (0/1 permission columns and a `label`
  column), it is used instead.
- **Account takeover.** Android apps cannot read Instagram or WhatsApp data, so this checker works
  only on simulated scenarios and no real messages are read. On the simulated logs it reaches
  about 94% on the test split (5-fold cross-validation 92.5%), but one scenario ("broadcaster") is
  still wrong in all 10 trials. In the app it is shown as a demo, next to a rule-based safety
  checklist that does not use machine learning.
- **Email checker.** The training set is very small and English only, so its answers are often
  close to 50% and should not be trusted.
- **QR checker.** It tries several image versions and two detectors. A picture that is not a valid
  QR code (for example a decorative placeholder) cannot be read.
- **Languages.** The app is translated into several languages as a beta. Translating the screen
  does not make the detectors understand other languages. The reports from the server stay in
  English.
- **Hosting.** The free Render tier has 512 MB of memory and sleeps when idle. Models are trained
  again after every restart. For APK files only the manifest is analysed to stay within memory.
