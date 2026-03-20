# NetGuard IDS — Streamlit App

## Files
```
streamlit_demo/
├── app.py            ← Streamlit app (frontend + backend in one)
├── requirements.txt  ← Python dependencies
├── model.joblib      ← PUT YOUR MODEL FILE HERE
└── README.md
```

## Run locally
```bash
pip install -r requirements.txt
py -m streamlit run app.py
```
Then open http://localhost:8501 in your browser.

---

## Deploy to Streamlit Community Cloud (free shareable link)

1. **Push to GitHub**
   - Create a new GitHub repo (can be private)
   - Upload `app.py`, `requirements.txt`, and `model.joblib` to the repo root

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click **"New app"**
   - Select your repo, branch (`main`), and set main file to `app.py`
   - Click **Deploy** — takes ~2 minutes

3. **Share the link**
   - You'll get a URL like: `https://your-app-name.streamlit.app`
   - Share this with your reviewer!

---

## Notes
- If `model.joblib` is missing, the app runs in **demo mode** automatically
  (simulated predictions) so the UI still works for screen recording
- Model must output `0` for Normal and `1` for Attack
- `predict_proba()` is used for confidence score if available
