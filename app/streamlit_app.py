"""
Driver Monitoring System v2 — Streamlit App
New: Yawning detection, Hand-to-ear phone detection,
     Safety score, Better alerts
Cairo University | FCAI | AI Department | 2026
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from PIL import Image

from models.drowsiness_detector import DrowsinessDetector
from models.object_detector     import ObjectDetector
from models.alert_manager       import AlertManager
from models.safety_score        import SafetyScore

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Monitoring System v2",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .alert-box {
        background: #ff4444; color: white;
        padding: 10px 16px; border-radius: 8px;
        font-size: 18px; font-weight: bold;
        text-align: center;
    }
    .warn-box {
        background: #ff8800; color: white;
        padding: 10px 16px; border-radius: 8px;
        font-size: 18px; font-weight: bold;
        text-align: center;
    }
    .safe-box {
        background: #00cc66; color: white;
        padding: 10px 16px; border-radius: 8px;
        font-size: 16px; text-align: center;
    }
    .score-box {
        padding: 16px; border-radius: 12px;
        text-align: center; font-size: 48px;
        font-weight: bold; color: white;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in {
    "running": False, "alert_mgr": None,
    "safety":  None,  "start_time": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    d = DrowsinessDetector()
    o = ObjectDetector(
        cigarette_model_path=os.path.join(
            os.path.dirname(__file__), "..", "models", "weights", "cigarette_model.pt"
        )
    )
    return d, o

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 Driver Monitor v2")
    st.caption("Cairo University — FCAI | AI | 2026")
    st.divider()

    st.subheader("⚙️ Detection")
    det_drowsy = st.toggle("😴 Drowsiness",  value=True)
    det_yawn   = st.toggle("😮 Yawning",     value=True)
    det_phone  = st.toggle("📱 Phone to Ear", value=True)
    det_cig    = st.toggle("🚬 Cigarette",    value=True)

    st.divider()
    st.subheader("🎚️ Thresholds")
    ear_thresh = st.slider("EAR (Drowsiness)", 0.15, 0.35, 0.25, 0.01)
    mar_thresh = st.slider("MAR (Yawning)",    0.40, 0.80, 0.60, 0.01)
    conf_thresh= st.slider("YOLO Confidence",  0.30, 0.90, 0.45, 0.05)

    st.divider()
    st.subheader("🔊 Alerts")
    sound_on   = st.toggle("Sound Alerts", value=True)
    cooldown   = st.slider("Alert Cooldown (sec)", 1, 10, 3)

    st.divider()
    stats_ph = st.empty()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🚗 Driver Monitoring System v2")
st.caption("Drowsiness · Yawning · Phone-to-Ear · Cigarette Detection")

tab_live, tab_upload, tab_about = st.tabs(["🎥 Live Monitor", "📁 Analyze File", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    # Layout
    col_feed, col_status = st.columns([3, 1])

    with col_feed:
        frame_ph  = st.empty()
        status_ph = st.empty()

    with col_status:
        st.subheader("📊 Status")
        drow_ph  = st.empty()
        yawn_ph  = st.empty()
        phone_ph = st.empty()
        cig_ph   = st.empty()
        st.divider()

        # Safety Score
        st.subheader("🏆 Safety Score")
        score_ph = st.empty()
        grade_ph = st.empty()
        st.divider()

        # EAR/MAR charts
        st.subheader("📈 Eye & Mouth")
        ear_ph = st.empty()
        st.divider()
        alert_ph = st.empty()

    col1, col2, col3 = st.columns(3)
    with col1:
        start_btn = st.button("▶️ Start", type="primary",   use_container_width=True)
    with col2:
        stop_btn  = st.button("⏹️ Stop",  type="secondary", use_container_width=True)
    with col3:
        reset_btn = st.button("🔄 Reset Score", use_container_width=True)

    if start_btn:
        st.session_state.running    = True
        st.session_state.start_time = time.time()
        st.session_state.alert_mgr  = AlertManager()
        st.session_state.safety     = SafetyScore()
        st.session_state.alert_mgr.cooldown = cooldown

    if stop_btn:
        st.session_state.running = False

    if reset_btn and st.session_state.safety:
        st.session_state.safety.reset()

    # ── Detection loop ────────────────────────────────────────────────────
    if st.session_state.running:
        drowsy_det, obj_det = load_models()

        # Apply thresholds
        import models.drowsiness_detector as dd
        dd.EAR_THRESHOLD = ear_thresh
        dd.MAR_THRESHOLD = mar_thresh
        import models.object_detector as od
        od.CONF_THRESHOLD = conf_thresh

        alert_mgr = st.session_state.alert_mgr
        safety    = st.session_state.safety
        alert_mgr.cooldown = cooldown

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot open webcam!")
            st.session_state.running = False
        else:
            ear_hist = []
            mar_hist = []

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.flip(frame, 1)

                # ── Run detectors ─────────────────────────────────────
                d_res = drowsy_det.detect(frame)
                frame = d_res["frame"]
                o_res = obj_det.detect(frame)
                frame = o_res["frame"]

                # ── Process alerts ────────────────────────────────────
                if det_drowsy and d_res["drowsy"]:
                    if alert_mgr.trigger("drowsiness", "Drowsy!", sound_on):
                        safety.deduct("drowsiness")

                if det_yawn and d_res["yawning"]:
                    if alert_mgr.trigger("yawning", "Yawning!", sound_on):
                        safety.deduct("yawning")

                if det_phone and o_res["phone_detected"]:
                    if alert_mgr.trigger("phone", "Phone!", sound_on):
                        safety.deduct("phone")

                if det_cig and o_res["cigarette_detected"]:
                    if alert_mgr.trigger("cigarette", "Cigarette!", sound_on):
                        safety.deduct("cigarette")

                # ── Update display ────────────────────────────────────
                frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               channels="RGB", use_container_width=True)

                # Status indicators
                drow_ph.markdown(
                    '<div class="alert-box">😴 DROWSY!</div>'    if (det_drowsy and d_res["drowsy"])   else
                    '<div class="safe-box">👁️ Alert</div>',
                    unsafe_allow_html=True)

                yawn_ph.markdown(
                    '<div class="warn-box">😮 YAWNING!</div>'   if (det_yawn and d_res["yawning"])    else
                    f'<div class="safe-box">😐 Yawns: {d_res["yawn_count"]}</div>',
                    unsafe_allow_html=True)

                phone_ph.markdown(
                    '<div class="alert-box">📱 PHONE!</div>'    if (det_phone and o_res["phone_detected"])   else
                    '<div class="safe-box">📵 No Phone</div>',
                    unsafe_allow_html=True)

                cig_ph.markdown(
                    '<div class="alert-box">🚬 SMOKING!</div>'  if (det_cig and o_res["cigarette_detected"]) else
                    '<div class="safe-box">✅ No Smoke</div>',
                    unsafe_allow_html=True)

                # Safety score
                s = safety.get_stats()
                score_ph.markdown(
                    f'<div class="score-box" style="background:#{s["color"]}">{s["score"]}</div>',
                    unsafe_allow_html=True)
                grade_ph.markdown(
                    f'<div style="text-align:center;font-size:20px;font-weight:bold">'
                    f'Grade: {s["grade"]} — {s["label"]}</div>',
                    unsafe_allow_html=True)

                # EAR/MAR chart
                ear_hist.append(d_res["ear"])
                mar_hist.append(d_res["mar"])
                if len(ear_hist) > 60: ear_hist.pop(0)
                if len(mar_hist) > 60: mar_hist.pop(0)

                chart_df = pd.DataFrame({"EAR": ear_hist, "MAR": mar_hist})
                ear_ph.line_chart(chart_df, height=150)

                # Alert log
                stats = alert_mgr.get_stats()
                log   = "\n".join(
                    [f"**{a['time']}** {a['type'].upper()}" for a in reversed(stats["log"])]
                ) or "No alerts"
                alert_ph.markdown(log)

                # Sidebar stats
                elapsed = int(time.time() - st.session_state.start_time)
                stats_ph.markdown(f"""
| | |
|--|--|
| ⏱️ | {elapsed}s |
| 😴 | {stats['drowsiness']} drowsy |
| 😮 | {stats.get('yawning',0)} yawns |
| 📱 | {stats['phone']} phone |
| 🚬 | {stats['cigarette']} smoke |
| 🏆 | {s['score']} / 100 |
""")

            cap.release()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader("📁 Analyze Image or Video")
    uploaded = st.file_uploader("Upload file", type=["jpg","jpeg","png","mp4","avi","mov"])

    if uploaded:
        drowsy_det, obj_det = load_models()
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)

        if uploaded.type.startswith("image"):
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            d_res = drowsy_det.detect(frame.copy())
            o_res = obj_det.detect(d_res["frame"])
            result = o_res["frame"]

            with col2:
                st.markdown("**Detection Result**")
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("😴 Drowsiness", "DETECTED" if d_res["drowsy"]              else "Clear")
            c2.metric("😮 Yawning",    "DETECTED" if d_res["yawning"]             else "Clear")
            c3.metric("📱 Phone",      "DETECTED" if o_res["phone_detected"]       else "Clear")
            c4.metric("🚬 Cigarette",  "DETECTED" if o_res["cigarette_detected"]   else "Clear")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
## 🚗 Driver Monitoring System v2

**Cairo University — Faculty of Computers & Artificial Intelligence**
**Department: Artificial Intelligence | 2025–2026**

### 🆕 New in v2
| Feature | Method |
|---------|--------|
| 😮 Yawning Detection | Mouth Aspect Ratio (MAR) via MediaPipe |
| 📱 Phone-to-Ear Detection | Hand + Ear proximity via MediaPipe Hands |
| 🏆 Driver Safety Score | Real-time 0–100 score with grade |
| 🔊 Smart Alerts | Cooldown-based, no false alarms |

### 🎯 All Detection Methods
| Hazard | Method | Model |
|--------|--------|-------|
| 😴 Drowsiness | EAR < 0.25 for 20 frames | MediaPipe FaceMesh |
| 😮 Yawning | MAR > 0.60 for 15 frames | MediaPipe FaceMesh |
| 📱 Phone Usage | Hand within 18% of frame from ear | MediaPipe Hands |
| 🚬 Cigarette | Object Detection | YOLOv8 fine-tuned |
""")
