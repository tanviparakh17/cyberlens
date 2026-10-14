# CyberLens - Project Build Notes (backend)

These notes record how the backend was built, what was measured, and what is still weak.
The README has the short version; this file has the history.

## 1. Setup and data

- Folders: `agents/` (one module per checker), `utils/` (shared helpers), `models/` (trained
  models, not in git), `data/` (datasets).
- Phishing URLs: Kaggle "Phishing Site URLs" (`phishing_site_urls.csv`, 156,422 rows labelled bad).
- Safe URLs: Tranco top-1M list (`top-1m.csv`, 1,000,000 domains).
- Both files are too big for git. They are shared separately (for example on Google Drive) and are
  only needed to rebuild the training data.
- `prepare_data.py` builds `data/phishing_urls.csv` (columns `url`, `label`; 0 = safe, 1 = phishing).
- `.gitignore`: venv, the big CSV files and the trained models.

### Team onboarding
1. `git clone` the repository.
2. `python -m venv venv`, activate it, `pip install -r requirements.txt`.
3. Put the two big CSV files into `data/` only if you need to rebuild the training data.
4. Run `uvicorn api:app --reload --port 8000` and open `http://localhost:8000/docs`.
5. Git workflow: add files by name (not `git add .`), commit, push. Do not commit `test_files/`.

## 2. Pipeline

`agent.predict()` -> `xai_agent` (SHAP, top 5 features) -> `report_agent` (text report, times in IST)
-> FastAPI JSON. The Flutter app calls the API and shows one result screen for every checker.

## 3. Deployment

- Render free tier: https://cyberlens-api-ukku.onrender.com (docs at `/docs`).
- The first request after idle can take 50+ seconds (cold start). The trained models are not in
  the repository, so they are trained again on first use after every restart.
- The free tier has 512 MB of memory. An APK upload therefore only reads the manifest (see 4.3).

## 4. Checkers: what was built and what was measured

### 4.1 URL checker
- 6,010 training URLs (3,005 phishing, 3,005 safe). Safe URLs include synthetic ones with
  realistic long paths (about 1,600 of them), and both classes are sampled to the same length
  distribution (mean about 45 characters for safe and 47 for phishing).
- Why: the first version learned a shortcut. Phishing URLs in the raw data were much longer than
  safe ones (mean 68 vs 43), and `url_length` was the strongest feature. The model looked
  accurate (0.87) but flagged normal long links, for example a restaurant menu link.
- Features (26): hostname and keyword features (hostname length, dots, hyphens, `@`, IP address,
  subdomains, suspicious keywords, brand in subdomain, redirects, shortened links), brand
  impersonation features (look-alike characters, hyphenated brand names, suspicious endings,
  digits mixed into words) and 10 URL-shape features (path depth and length, query length and
  parameters, digit ratio, host and path entropy, longest host label, special-character ratio).
  Raw `url_length` and `num_digits` were removed on purpose.
- Experiment (`url_experiment.py`, 4 variants, not committed): the shape features without raw
  length scored 0.786 on 3,000 real URLs whose domains were not in the training data, against
  0.736 for the old features. Phishing recall rose from 0.77 to 0.86; safe recall stayed near 0.71.
- Result of the final model: domain-separated test accuracy 0.851 (1,798 URLs).
- On top of the model: a rule layer (brand impersonation, IP host with a suspicious keyword,
  suspicious ending with a suspicious keyword), an allowlist (`trusted_domains.txt` from the top
  Tranco domains, `known_good_domains.txt`, official brand domains), and a "suspicious" verdict when
  the model alone says phishing with less than 70% confidence.
- Hand-made check (`url_fp_check.py`, 8 safe and 6 phishing links, not committed): 0 of 8 safe links
  flagged, 6 of 6 phishing links caught, but all 6 by the rules, not by the model.
- Weak points: `spot_check.py` (21 links) was used while tuning the rules, so it is a smoke test.
  Safe recall on real unseen URLs is only about 71%. The college site is allowlisted (demo tuning).
  A new legitimate domain can still be flagged.
- Helper scripts: `check_data.py` prints URL length by class and how well each feature separates
  the classes (this is how the length shortcut was found); `holdout_check.py` runs the URL agent on
  hand-written links that are not in the training data.
- A URL with an unmatched `[` made the standard `urlparse` raise an error and crash training. A
  safe `urlparse` was added in `utils/feature_extraction.py`.

### 4.2 Email checker
TF-IDF words plus a few heuristics (links, exclamation marks, dollar signs, length, urgency words),
RandomForest. The training file is tiny (10 emails when checked), the language is English only, and
answers often sit near 50%. Treat it as a demo.

### 4.3 APK checker
- Androguard 4.1.4. The API reads only the manifest (permissions); the full code analysis is only
  used from the command line (`python -m agents.apk_agent file.apk`).
- Measured on a 16.4 MB APK: manifest-only 3.6 s and 330 MB peak memory, full analysis 11.8 s and
  1,113 MB peak. The full analysis caused 502 errors on Render, so the API uses the light mode.
- Model: RandomForest on permissions with SHAP. There was no labelled APK dataset, so it is trained
  on synthetic permission profiles. If `data/apk_permissions.csv` exists it is used instead.
  Verdicts: below 0.35 safe, below 0.70 suspicious, otherwise malicious.
- Upload time (not analysis time) dominates the request on a slow connection.

### 4.4 QR checker
Decodes the QR code with OpenCV and sends the link to the URL checker. If the plain detector fails
it tries shrunk, enlarged, bordered, blurred and thresholded copies and a second detector, with an
8 second limit. A picture that is not a valid QR code cannot be read.

### 4.5 Account takeover checker
- Graph features (networkx) from message logs, RandomForest. The logs are simulated
  (`generate_ato_logs.py`: 800 accounts, 138,749 messages). About 94% on the test split, ROC-AUC 0.99,
  5-fold cross-validation 92.5%.
- Weakness: the "broadcaster" scenario is wrong in all 10 trials.
- An Android app cannot read Instagram or WhatsApp data, so in the app this is a demo on four
  simulated scenarios. The app also has a rule-based "was my account hacked" checklist, which is not
  machine learning.
- Verdicts are `normal` and `suspicious`.

## 5. Not verified yet
- A 25 MB APK on Render (the limit in the API).
- QR photos taken with a phone camera, on Render.
- Time of the first request after a restart, when the models are trained again.
