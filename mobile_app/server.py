"""
Driver Monitoring System — Flask Mobile Server
Phone camera → Flask → AI Models → Results back to phone
Cairo University | FCAI | AI Department | 2026
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import time
import threading

from models.drowsiness_detector import DrowsinessDetector
from models.object_detector     import ObjectDetector
from models.alert_manager       import AlertManager
from models.safety_score        import SafetyScore

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dms-cairo-university-2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── Load models once ──────────────────────────────────────────────────────────
print("[Server] Loading AI models...")
drowsy_det  = DrowsinessDetector()
object_det  = ObjectDetector(
    cigarette_model_path=os.path.join(
        os.path.dirname(__file__), "..", "models", "weights", "cigarette_model.pt"
    )
)
alert_mgr   = AlertManager()
safety      = SafetyScore()
print("[Server] Models loaded! ✅")

# ── Session stats ─────────────────────────────────────────────────────────────
session_stats = {
    "drowsiness": 0, "yawning": 0,
    "phone": 0,      "cigarette": 0,
    "score": 100,    "grade": "A",
    "frames": 0,     "start_time": time.time()
}

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stats')
def stats():
    return jsonify(session_stats)

@app.route('/reset')
def reset():
    global session_stats
    alert_mgr.reset()
    safety.reset()
    session_stats = {
        "drowsiness": 0, "yawning": 0,
        "phone": 0,      "cigarette": 0,
        "score": 100,    "grade": "A",
        "frames": 0,     "start_time": time.time()
    }
    return jsonify({"status": "reset"})

# ── SocketIO frame processing ──────────────────────────────────────────────────
@socketio.on('frame')
def handle_frame(data):
    global session_stats
    try:
        # Decode base64 frame from phone
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr    = np.frombuffer(img_data, np.uint8)
        frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        session_stats["frames"] += 1

        # ── Run detectors ─────────────────────────────────────────────────
        d_res = drowsy_det.detect(frame)
        frame = d_res["frame"]
        o_res = object_det.detect(frame)
        frame = o_res["frame"]

        # ── Process alerts ────────────────────────────────────────────────
        alerts = []

        if d_res["drowsy"]:
            if alert_mgr.trigger("drowsiness", "Drowsy!", True):
                safety.deduct("drowsiness")
                session_stats["drowsiness"] += 1
                alerts.append("😴 DROWSINESS ALERT!")

        if d_res["yawning"]:
            if alert_mgr.trigger("yawning", "Yawning!", True):
                safety.deduct("yawning")
                session_stats["yawning"] += 1
                alerts.append("😮 YAWNING DETECTED!")

        if o_res["phone_detected"]:
            if alert_mgr.trigger("phone", "Phone!", True):
                safety.deduct("phone")
                session_stats["phone"] += 1
                alerts.append("📱 PHONE TO EAR!")

        if o_res["cigarette_detected"]:
            if alert_mgr.trigger("cigarette", "Smoking!", True):
                safety.deduct("cigarette")
                session_stats["cigarette"] += 1
                alerts.append("🚬 SMOKING!")

        s = safety.get_stats()
        session_stats["score"] = s["score"]
        session_stats["grade"] = s["grade"]

        # ── Encode result frame ───────────────────────────────────────────
        _, buffer  = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        result_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

        # ── Send results back to phone ────────────────────────────────────
        emit('result', {
            'image':     result_b64,
            'drowsy':    d_res["drowsy"],
            'yawning':   d_res["yawning"],
            'phone':     o_res["phone_detected"],
            'cigarette': o_res["cigarette_detected"],
            'ear':       d_res["ear"],
            'mar':       d_res["mar"],
            'yawns':     d_res["yawn_count"],
            'score':     s["score"],
            'grade':     s["grade"],
            'color':     s["color"],
            'alerts':    alerts,
        })

    except Exception as e:
        print(f"[Server] Frame error: {e}")

@socketio.on('connect')
def on_connect():
    print(f"[Server] Phone connected!")

@socketio.on('disconnect')
def on_disconnect():
    print(f"[Server] Phone disconnected.")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("\n" + "="*55)
    print("  🚗 Driver Monitoring System — Mobile Server")
    print("  Cairo University | FCAI | AI Dept | 2026")
    print("="*55)
    print(f"\n  Open on your phone browser:")
    print(f"  👉  http://{local_ip}:5000")
    print(f"\n  Make sure phone and laptop are on same WiFi!")
    print("="*55 + "\n")

    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
