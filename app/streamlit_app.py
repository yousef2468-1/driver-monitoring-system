"""
Driver Monitoring System — Streamlit App
Cairo University | FCAI | AI Department | 2026
"""

import sys
import os
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


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Monitoring System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .alert-box {
        background: #ff4444;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        animation: pulse 1s infinite;
    }
    .safe-box {
        background: #00cc66;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 18px;
        text-align: center;
    }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    @keyframes pulse {
        0%   { opacity: 1.0; }
        50%  { opacity: 0.7; }
        100% { opacity: 1.0; }
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "running"        not in st.session_state: st.session_state.running        = False
if "alert_manager"  not in st.session_state: st.session_state.alert_manager  = AlertManager()
if "start_time"     not in st.session_state: st.session_state.start_time     = None
if "frame_count"    not in st.session_state: st.session_state.frame_count    = 0
if "drowsy_model"   not in st.session_state: st.session_state.drowsy_model   = None
if "object_model"   not in st.session_state: st.session_state.object_model   = None


# ─────────────────────────────────────────────────────────────────────────────
# Load models (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    drowsiness = DrowsinessDetector()
    objects    = ObjectDetector(
        cigarette_model_path=os.path.join(
            os.path.dirname(__file__), "..", "models", "weights", "cigarette.pt"
        )
    )
    return drowsiness, objects


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/c5/Cairo_University_Crest.svg/180px-Cairo_University_Crest.svg.png", width=80)
    st.title("🚗 Driver Monitor")
    st.caption("Cairo University — FCAI | AI Dept | 2026")
    st.divider()

    st.subheader("⚙️ Detection Settings")
    detect_drowsiness = st.toggle("Drowsiness Detection",  value=True)
    detect_phone      = st.toggle("Phone Usage Detection", value=True)
    detect_cigarette  = st.toggle("Cigarette Detection",   value=True)

    st.divider()
    st.subheader("🎚️ Thresholds")
    ear_thresh   = st.slider("EAR Threshold (Drowsiness)", 0.15, 0.35, 0.25, 0.01)
    conf_thresh  = st.slider("YOLO Confidence",            0.30, 0.90, 0.45, 0.05)

    st.divider()
    input_source = st.radio("📷 Input Source", ["Webcam", "Upload Video", "Upload Image"])

    st.divider()
    st.subheader("📊 Session Stats")
    stats_placeholder = st.empty()


# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────
st.title("🚗 Driver Monitoring System")
st.caption("Real-time detection of drowsiness · phone usage · cigarette smoking")

tab_live, tab_upload, tab_about = st.tabs(["🎥 Live Monitor", "📁 Analyze File", "ℹ️ About"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    col_video, col_status = st.columns([3, 1])

    with col_video:
        frame_window   = st.empty()
        status_window  = st.empty()

    with col_status:
        st.subheader("🔴 Status")
        drow_status  = st.empty()
        phone_status = st.empty()
        cig_status   = st.empty()
        st.divider()
        st.subheader("📈 EAR Value")
        ear_chart    = st.empty()
        st.divider()
        alert_log_ph = st.empty()

    col_start, col_stop = st.columns(2)
    with col_start:
        start_btn = st.button("▶️ Start Monitoring", type="primary",  use_container_width=True)
    with col_stop:
        stop_btn  = st.button("⏹️ Stop",             type="secondary", use_container_width=True)

    if start_btn:
        st.session_state.running      = True
        st.session_state.start_time   = time.time()
        st.session_state.frame_count  = 0
        st.session_state.alert_manager.reset()

    if stop_btn:
        st.session_state.running = False

    # ── Main detection loop ───────────────────────────────────────────────────
    if st.session_state.running:
        drowsiness_det, object_det = load_models()
        # override thresholds from sidebar
        from models import drowsiness_detector as dd_mod
        dd_mod.EAR_THRESHOLD = ear_thresh
        from models import object_detector as od_mod
        od_mod.CONF_THRESHOLD = conf_thresh

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot open webcam. Try the 'Analyze File' tab instead.")
            st.session_state.running = False
        else:
            ear_history = []
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)   # mirror
                alerts_this_frame = []

                # Drowsiness
                if detect_drowsiness:
                    d_res = drowsiness_det.detect(frame)
                    frame = d_res["frame"]
                    if d_res["drowsy"]:
                        alerts_this_frame.append("drowsiness")
                        st.session_state.alert_manager.trigger("drowsiness", "Driver drowsy!")
                    ear_history.append(d_res["ear"])
                    if len(ear_history) > 60:
                        ear_history.pop(0)
                    drow_status.markdown(
                        '<div class="alert-box">😴 DROWSY!</div>'   if d_res["drowsy"] else
                        '<div class="safe-box">👁️ Alert</div>',
                        unsafe_allow_html=True
                    )

                # Object (phone + cigarette)
                if detect_phone or detect_cigarette:
                    o_res = object_det.detect(frame)
                    frame = o_res["frame"]
                    if o_res["phone_detected"] and detect_phone:
                        alerts_this_frame.append("phone")
                        st.session_state.alert_manager.trigger("phone", "Phone usage!")
                    if o_res["cigarette_detected"] and detect_cigarette:
                        alerts_this_frame.append("cigarette")
                        st.session_state.alert_manager.trigger("cigarette", "Smoking detected!")

                    phone_status.markdown(
                        '<div class="alert-box">📱 PHONE!</div>'  if (o_res["phone_detected"] and detect_phone) else
                        '<div class="safe-box">📵 No Phone</div>',
                        unsafe_allow_html=True
                    )
                    cig_status.markdown(
                        '<div class="alert-box">🚬 SMOKING!</div>' if (o_res["cigarette_detected"] and detect_cigarette) else
                        '<div class="safe-box">✅ No Smoke</div>',
                        unsafe_allow_html=True
                    )

                # Show frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, channels="RGB", use_container_width=True)

                # EAR mini chart
                if ear_history:
                    ear_df = pd.DataFrame({"EAR": ear_history})
                    ear_chart.line_chart(ear_df, height=120)

                # Alert log
                stats = st.session_state.alert_manager.get_stats()
                log_md = "\n".join(
                    [f"**{a['time']}** — {a['type'].upper()}" for a in reversed(stats["log"])]
                ) or "No alerts yet"
                alert_log_ph.markdown(log_md)

                # Sidebar stats
                elapsed = int(time.time() - st.session_state.start_time)
                stats_placeholder.markdown(f"""
| Metric | Count |
|--------|-------|
| ⏱️ Time | {elapsed}s |
| 😴 Drowsy | {stats['drowsiness']} |
| 📱 Phone | {stats['phone']} |
| 🚬 Smoke | {stats['cigarette']} |
""")
                st.session_state.frame_count += 1

            cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD / ANALYZE
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader("📁 Analyze an Image or Video")
    uploaded = st.file_uploader("Upload image or video", type=["jpg","jpeg","png","mp4","avi","mov"])

    if uploaded:
        drowsiness_det, object_det = load_models()
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)

        if uploaded.type.startswith("image"):
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original**")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Run all detectors
            d_res = drowsiness_det.detect(frame.copy())
            o_res = object_det.detect(d_res["frame"])
            result_frame = o_res["frame"]

            with col2:
                st.markdown("**Detection Result**")
                st.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Results
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("😴 Drowsiness", "DETECTED" if d_res["drowsy"]              else "Clear", delta=None)
            c2.metric("📱 Phone",      "DETECTED" if o_res["phone_detected"]       else "Clear", delta=None)
            c3.metric("🚬 Cigarette",  "DETECTED" if o_res["cigarette_detected"]   else "Clear", delta=None)

        elif uploaded.type.startswith("video"):
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(file_bytes.tobytes())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            stframe = st.empty()
            progress = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0
            alert_mgr = AlertManager()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                d_res = drowsiness_det.detect(frame.copy())
                o_res = object_det.detect(d_res["frame"])
                result = o_res["frame"]
                stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
                frame_idx += 1
                progress.progress(min(frame_idx / max(total_frames, 1), 1.0))

            cap.release()
            st.success("✅ Video analysis complete!")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
## 🚗 Driver Monitoring System

**Cairo University — Faculty of Computers & Artificial Intelligence**
**Department: Artificial Intelligence | Academic Year: 2025–2026**

---

### 🎯 Project Overview
A real-time driver safety system that uses computer vision and deep learning to detect:

| Hazard | Method | Model |
|--------|--------|-------|
| 😴 Drowsiness | Eye Aspect Ratio (EAR) | MediaPipe FaceMesh |
| 📱 Phone Usage | Object Detection | YOLOv8n (COCO) |
| 🚬 Cigarette Smoking | Object Detection | YOLOv8 (fine-tuned) |

---

### 🏗️ System Architecture
```
Camera Input → Frame Preprocessing → Parallel Detection Pipeline
                                           ├── MediaPipe (Drowsiness)
                                           ├── YOLOv8 (Phone)
                                           └── YOLOv8 (Cigarette)
                                      ↓
                               Alert Manager → Audio + Visual Alerts
                                      ↓
                               Streamlit Dashboard
```

---

### 📚 Datasets Used
- **MRL Eye Dataset** — 84,898 eye images for drowsiness research
- **Roboflow Smoking Detection** — ~2,000 annotated cigarette images
- **COCO Dataset** — 330,000 images including cell phone class

---

### 👥 Team
*Cairo University | FCAI | AI Department | 2025–2026*
""")
