import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="KAGE DIGITAL DEFENSE",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@400;500;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.75rem 1.5rem;
    background: #131614;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.logo-name { font-size: 22px; font-weight: 800; color: #e8ede9; letter-spacing: -0.5px; }
.logo-sub  { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #7a8c80; }
.pill {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600;
    color: #1D9E75; background: rgba(29,158,117,0.12);
    border: 1px solid rgba(29,158,117,0.3);
    padding: 5px 14px; border-radius: 20px;
}

.result-normal {
    background: rgba(29,158,117,0.1); border: 1px solid rgba(29,158,117,0.35);
    border-radius: 12px; padding: 1.5rem; text-align: center;
}
.result-attack {
    background: rgba(226,75,74,0.1); border: 1px solid rgba(226,75,74,0.35);
    border-radius: 12px; padding: 1.5rem; text-align: center;
}
.result-idle {
    background: #1e2220; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 1.5rem; text-align: center;
}
.verdict-normal { font-size: 38px; font-weight: 800; color: #1D9E75; letter-spacing: -1.5px; }
.verdict-attack { font-size: 38px; font-weight: 800; color: #E24B4A; letter-spacing: -1.5px; }
.verdict-idle   { font-size: 38px; font-weight: 800; color: #7a8c80; letter-spacing: -1.5px; }
.verdict-label  { font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px; }

.stat-card {
    background: #1e2220; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 0.9rem 1.1rem; text-align: center;
}
.stat-num-safe   { font-size: 28px; font-weight: 800; color: #1D9E75; }
.stat-num-danger { font-size: 28px; font-weight: 800; color: #E24B4A; }
.stat-desc { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #7a8c80; margin-top: 2px; }

.section-label {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase; color: #7a8c80; margin-bottom: 8px;
}
.log-item-normal {
    background: #1e2220; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 7px; padding: 7px 12px; margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    display: flex; justify-content: space-between;
    color: #1D9E75;
}
.log-item-attack {
    background: #1e2220; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 7px; padding: 7px 12px; margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    display: flex; justify-content: space-between;
    color: #E24B4A;
}
.stNumberInput label { font-family: 'JetBrains Mono', monospace !important; font-size: 10px !important; color: #7a8c80 !important; }
div[data-testid="stNumberInput"] input { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ── session state ──────────────────────────────────────────────
if "log"          not in st.session_state: st.session_state.log = []
if "total_normal" not in st.session_state: st.session_state.total_normal = 0
if "total_attack" not in st.session_state: st.session_state.total_attack = 0
if "result"       not in st.session_state: st.session_state.result = None

# ── load model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.environ.get("MODEL_PATH", "model.joblib")
    if os.path.exists(path):
        return joblib.load(path), True
    return None, False

model, model_loaded = load_model()

# ── feature definitions ────────────────────────────────────────
FEATURES = [
    ("min_seg_size_forward",        "Min Seg Size Fwd",       20.0),
    ("Destination Port",            "Destination Port",       80.0),
    ("Flow Duration",               "Flow Duration (µs)",     1234567.0),
    ("Total Length of Fwd Packets", "Total Fwd Pkt Length",   4800.0),
    ("Total Length of Bwd Packets", "Total Bwd Pkt Length",   12200.0),
    ("Fwd Packet Length Min",       "Fwd Pkt Len Min",        40.0),
    ("Fwd Packet Length Mean",      "Fwd Pkt Len Mean",       120.0),
    ("Bwd Packet Length Min",       "Bwd Pkt Len Min",        40.0),
    ("Flow Bytes/s",                "Flow Bytes/s",           13780.0),
    ("Flow Packets/s",              "Flow Packets/s",         45.0),
    ("Flow IAT Min",                "Flow IAT Min (µs)",      1200.0),
    ("Bwd IAT Total",               "Bwd IAT Total (µs)",     850000.0),
    ("Bwd IAT Mean",                "Bwd IAT Mean (µs)",      70000.0),
    ("Bwd IAT Std",                 "Bwd IAT Std (µs)",       12000.0),
    ("Bwd Packets/s",               "Bwd Packets/s",          22.0),
    ("Min Packet Length",           "Min Packet Length",      40.0),
    ("Packet Length Mean",          "Packet Length Mean",     80.0),
    ("FIN Flag Count",              "FIN Flag Count",         1.0),
    ("SYN Flag Count",              "SYN Flag Count",         1.0),
    ("PSH Flag Count",              "PSH Flag Count",         1.0),
    ("ACK Flag Count",              "ACK Flag Count",         1.0),
    ("URG Flag Count",              "URG Flag Count",         0.0),
    ("Down/Up Ratio",               "Down/Up Ratio",          2.5),
    ("Init_Win_bytes_forward",      "Init Win Bytes Fwd",     65535.0),
    ("Init_Win_bytes_backward",     "Init Win Bytes Bwd",     65535.0),
    ("Active Mean",                 "Active Mean (µs)",       145000.0),
    ("Active Std",                  "Active Std (µs)",        23000.0),
    ("Idle Std",                    "Idle Std (µs)",          0.0),
]

PRESETS = {
    "Normal Traffic": {
        "min_seg_size_forward":20,"Destination Port":80,"Flow Duration":1234567,
        "Total Length of Fwd Packets":4800,"Total Length of Bwd Packets":12200,
        "Fwd Packet Length Min":40,"Fwd Packet Length Mean":120,"Bwd Packet Length Min":40,
        "Flow Bytes/s":13780,"Flow Packets/s":45,"Flow IAT Min":1200,
        "Bwd IAT Total":850000,"Bwd IAT Mean":70000,"Bwd IAT Std":12000,"Bwd Packets/s":22,
        "Min Packet Length":40,"Packet Length Mean":80,
        "FIN Flag Count":1,"SYN Flag Count":1,"PSH Flag Count":1,"ACK Flag Count":1,"URG Flag Count":0,
        "Down/Up Ratio":2.5,"Init_Win_bytes_forward":65535,"Init_Win_bytes_backward":65535,
        "Active Mean":145000,"Active Std":23000,"Idle Std":0,
    },
    "Port Scan Attack": {
        "min_seg_size_forward":20,"Destination Port":22,"Flow Duration":12000,
        "Total Length of Fwd Packets":40,"Total Length of Bwd Packets":0,
        "Fwd Packet Length Min":40,"Fwd Packet Length Mean":40,"Bwd Packet Length Min":0,
        "Flow Bytes/s":3333,"Flow Packets/s":83,"Flow IAT Min":0,
        "Bwd IAT Total":0,"Bwd IAT Mean":0,"Bwd IAT Std":0,"Bwd Packets/s":0,
        "Min Packet Length":40,"Packet Length Mean":40,
        "FIN Flag Count":0,"SYN Flag Count":1,"PSH Flag Count":0,"ACK Flag Count":0,"URG Flag Count":0,
        "Down/Up Ratio":0,"Init_Win_bytes_forward":1024,"Init_Win_bytes_backward":0,
        "Active Mean":0,"Active Std":0,"Idle Std":0,
    },
    "DoS Attack": {
        "min_seg_size_forward":20,"Destination Port":80,"Flow Duration":60000000,
        "Total Length of Fwd Packets":960000,"Total Length of Bwd Packets":0,
        "Fwd Packet Length Min":60,"Fwd Packet Length Mean":60,"Bwd Packet Length Min":0,
        "Flow Bytes/s":16000,"Flow Packets/s":266,"Flow IAT Min":50,
        "Bwd IAT Total":0,"Bwd IAT Mean":0,"Bwd IAT Std":0,"Bwd Packets/s":0,
        "Min Packet Length":60,"Packet Length Mean":60,
        "FIN Flag Count":0,"SYN Flag Count":1,"PSH Flag Count":0,"ACK Flag Count":0,"URG Flag Count":0,
        "Down/Up Ratio":0,"Init_Win_bytes_forward":8192,"Init_Win_bytes_backward":0,
        "Active Mean":0,"Active Std":0,"Idle Std":0,
    },
}

def demo_predict(values):
    d = dict(zip([f[0] for f in FEATURES], values))
    score = 0
    if d["SYN Flag Count"] > 0 and d["Total Length of Bwd Packets"] == 0: score += 40
    if d["Destination Port"] < 1024 and d["Bwd Packets/s"] == 0:          score += 20
    if d["Flow Bytes/s"] > 12000 and d["Total Length of Bwd Packets"] == 0: score += 25
    if d["Flow IAT Min"] < 100:                                             score += 10
    is_attack = score >= 45
    conf = round(np.clip(np.random.uniform(0.72, 0.97) if is_attack else np.random.uniform(0.70, 0.97), 0, 0.99), 4)
    return {"label": "ATTACK" if is_attack else "NORMAL", "is_attack": is_attack, "confidence": conf}

# ── topbar ─────────────────────────────────────────────────────
mode_text = "XGBoost — Live Model" if model_loaded else "Demo Mode"
st.markdown(f"""
<div class="topbar">
  <div>
    <div class="logo-name">🛡️ KDD-Kage Digital Defense</div>
    <div class="logo-sub">intrusion detection system · xgboost binary classifier</div>
  </div>
  <div class="pill">● {mode_text}</div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.warning("⚠️ `model.joblib` not found — running in demo mode with simulated predictions. Upload your model file to the repo root to enable live inference.", icon="⚠️")

# ── preset buttons ─────────────────────────────────────────────
st.markdown('<div class="section-label">Load a sample preset</div>', unsafe_allow_html=True)
preset_cols = st.columns(4)
chosen_preset = None
if preset_cols[0].button("Normal Traffic",   use_container_width=True): chosen_preset = "Normal Traffic"
if preset_cols[1].button("Port Scan Attack", use_container_width=True): chosen_preset = "Port Scan Attack"
if preset_cols[2].button("DoS Attack",       use_container_width=True): chosen_preset = "DoS Attack"
if preset_cols[3].button("Clear All",        use_container_width=True): chosen_preset = "__clear__"

# ── feature inputs ─────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:1rem">Flow & packet features (28)</div>', unsafe_allow_html=True)

def get_default(key):
    if chosen_preset == "__clear__": return 0.0
    if chosen_preset and chosen_preset in PRESETS: return float(PRESETS[chosen_preset][key])
    return next((f[2] for f in FEATURES if f[0] == key), 0.0)

cols_per_row = 4
rows = [FEATURES[i:i+cols_per_row] for i in range(0, len(FEATURES), cols_per_row)]
input_values = {}

for row in rows:
    cols = st.columns(cols_per_row)
    for i, (key, label, default) in enumerate(row):
        input_values[key] = cols[i].number_input(label, value=get_default(key), step=None, format="%g", key=f"feat_{key}_{chosen_preset}")

# ── analyze button ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
left_btn, _, right_btn = st.columns([2, 3, 2])
run = left_btn.button("🔍  Analyze Traffic", use_container_width=True, type="primary")

if run:
    values = np.array([input_values[f[0]] for f in FEATURES]).reshape(1, -1)
    if model_loaded:
        pred = int(model.predict(values)[0])
        is_attack = pred == 1
        conf = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(values)[0]
            conf = round(float(proba[pred]), 4)
        result = {"label": "ATTACK" if is_attack else "NORMAL", "is_attack": is_attack, "confidence": conf}
    else:
        result = demo_predict(values[0])

    st.session_state.result = result
    if result["is_attack"]: st.session_state.total_attack += 1
    else:                   st.session_state.total_normal += 1

    import datetime
    t = datetime.datetime.now().strftime("%H:%M:%S")
    conf_str = f"{result['confidence']*100:.1f}%" if result.get("confidence") else ""
    st.session_state.log.insert(0, {"label": result["label"], "is_attack": result["is_attack"], "conf": conf_str, "time": t})
    if len(st.session_state.log) > 30: st.session_state.log.pop()

# ── result + stats + log ───────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
res_col, stats_col = st.columns([3, 2])

with res_col:
    st.markdown('<div class="section-label">Classification result</div>', unsafe_allow_html=True)
    r = st.session_state.result
    if r is None:
        st.markdown('<div class="result-idle"><div class="verdict-label" style="color:#7a8c80">awaiting analysis</div><div class="verdict-idle">— — —</div><div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#7a8c80;margin-top:8px">Enter feature values and click Analyze Traffic</div></div>', unsafe_allow_html=True)
    elif r["is_attack"]:
        conf_html = f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#E24B4A;margin-top:8px">Confidence: {r["confidence"]*100:.1f}%</div>' if r.get("confidence") else ""
        st.markdown(f'<div class="result-attack"><div class="verdict-label" style="color:#E24B4A">Intrusion Detected</div><div class="verdict-attack">ATTACK</div>{conf_html}<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#E24B4A;margin-top:8px">Flagged — anomalous flow patterns detected</div></div>', unsafe_allow_html=True)
    else:
        conf_html = f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#1D9E75;margin-top:8px">Confidence: {r["confidence"]*100:.1f}%</div>' if r.get("confidence") else ""
        st.markdown(f'<div class="result-normal"><div class="verdict-label" style="color:#1D9E75">Benign Traffic</div><div class="verdict-normal">NORMAL</div>{conf_html}<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#1D9E75;margin-top:8px">Flow consistent with normal network behavior</div></div>', unsafe_allow_html=True)

with stats_col:
    st.markdown('<div class="section-label">Session stats</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="stat-card"><div class="stat-num-safe">{st.session_state.total_normal}</div><div class="stat-desc">normal flows</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-card"><div class="stat-num-danger">{st.session_state.total_attack}</div><div class="stat-desc">attacks detected</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.2rem">Detection log</div>', unsafe_allow_html=True)
    if not st.session_state.log:
        st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#7a8c80;text-align:center;padding:1rem 0">No classifications yet</div>', unsafe_allow_html=True)
    for entry in st.session_state.log[:10]:
        cls = "log-item-attack" if entry["is_attack"] else "log-item-normal"
        st.markdown(f'<div class="{cls}"><span>{entry["label"]} {entry["conf"]}</span><span style="color:#7a8c80">{entry["time"]}</span></div>', unsafe_allow_html=True)
