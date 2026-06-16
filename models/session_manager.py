"""
Session Manager
- Records annotated video
- Exports alert log as CSV
- Generates PDF session report
"""

import cv2
import csv
import os
import time
from datetime import datetime


class SessionManager:
    def __init__(self, output_dir="sessions"):
        self.output_dir   = output_dir
        self.session_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir  = os.path.join(output_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)

        self.recording    = False
        self.video_writer = None
        self.frame_count  = 0
        self.start_time   = None
        self.alert_log    = []

    def start_recording(self, frame_width=640, frame_height=480, fps=20):
        video_path = os.path.join(self.session_dir, "session.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            video_path, fourcc, fps, (frame_width, frame_height)
        )
        self.recording  = True
        self.start_time = time.time()
        print(f"[SessionManager] Recording to {video_path}")

    def write_frame(self, frame):
        if self.recording and self.video_writer:
            self.video_writer.write(frame)
            self.frame_count += 1

    def log_alert(self, alert_type, message, score):
        self.alert_log.append({
            "timestamp":  time.strftime("%H:%M:%S"),
            "elapsed":    int(time.time() - self.start_time) if self.start_time else 0,
            "type":       alert_type,
            "message":    message,
            "score":      score
        })

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        print(f"[SessionManager] Recording stopped. {self.frame_count} frames saved.")

    def export_csv(self):
        csv_path = os.path.join(self.session_dir, "alerts.csv")
        if not self.alert_log:
            return None
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp","elapsed","type","message","score"])
            writer.writeheader()
            writer.writerows(self.alert_log)
        print(f"[SessionManager] CSV saved to {csv_path}")
        return csv_path

    def generate_report(self, safety_stats):
        """Generate a simple text report"""
        report_path = os.path.join(self.session_dir, "report.txt")
        duration = int(time.time() - self.start_time) if self.start_time else 0

        drowsy_count = sum(1 for a in self.alert_log if a["type"] == "drowsiness")
        yawn_count   = sum(1 for a in self.alert_log if a["type"] == "yawning")
        phone_count  = sum(1 for a in self.alert_log if a["type"] == "phone")
        cig_count    = sum(1 for a in self.alert_log if a["type"] == "cigarette")

        with open(report_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("  DRIVER MONITORING SYSTEM — SESSION REPORT\n")
            f.write("  Cairo University | FCAI | AI Dept | 2026\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Session ID   : {self.session_id}\n")
            f.write(f"Date         : {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"Duration     : {duration}s\n")
            f.write(f"Total Frames : {self.frame_count}\n\n")
            f.write("─" * 50 + "\n")
            f.write("SAFETY SCORE\n")
            f.write("─" * 50 + "\n")
            f.write(f"Final Score  : {safety_stats['score']} / 100\n")
            f.write(f"Grade        : {safety_stats['grade']} — {safety_stats['label']}\n\n")
            f.write("─" * 50 + "\n")
            f.write("ALERT SUMMARY\n")
            f.write("─" * 50 + "\n")
            f.write(f"Drowsiness   : {drowsy_count} alerts\n")
            f.write(f"Yawning      : {yawn_count} alerts\n")
            f.write(f"Phone Usage  : {phone_count} alerts\n")
            f.write(f"Smoking      : {cig_count} alerts\n")
            f.write(f"Total        : {len(self.alert_log)} alerts\n\n")
            f.write("─" * 50 + "\n")
            f.write("ALERT LOG\n")
            f.write("─" * 50 + "\n")
            for a in self.alert_log:
                f.write(f"[{a['timestamp']}] {a['type'].upper():12} | Score: {a['score']}\n")

        print(f"[SessionManager] Report saved to {report_path}")
        return report_path

    def get_session_path(self):
        return self.session_dir
