"""
Cigarette & Phone Detection Module
Uses YOLOv8 pretrained / fine-tuned model
Detects: cigarette (smoking), mobile phone (distraction)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os


# ── Class labels we care about ────────────────────────────────────────────────
PHONE_CLASSES      = {"cell phone"}          # COCO label
CIGARETTE_CLASSES  = {"cigarette", "smoking", "smoke"}  # custom model labels

# ── Confidence threshold ──────────────────────────────────────────────────────
CONF_THRESHOLD = 0.45


class ObjectDetector:
    """
    Dual-purpose detector:
      - Phone usage  : uses YOLOv8n trained on COCO (cell phone class built-in)
      - Cigarette    : uses YOLOv8 fine-tuned on smoking dataset from Roboflow
    """

    def __init__(self, cigarette_model_path: str = None):
        """
        Args:
            cigarette_model_path: path to fine-tuned .pt file for cigarette detection.
                                  If None, falls back to COCO yolov8n (less accurate
                                  but still works for demo).
        """
        # Phone detector — COCO model has 'cell phone' built in
        print("[ObjectDetector] Loading phone detector (YOLOv8n COCO)...")
        self.phone_model = YOLO("yolov8n.pt")   # auto-downloads on first run

        # Cigarette detector
        if cigarette_model_path and os.path.exists(cigarette_model_path):
            print(f"[ObjectDetector] Loading cigarette model from {cigarette_model_path}")
            self.cig_model = YOLO(cigarette_model_path)
        else:
            print("[ObjectDetector] No fine-tuned cigarette model found — using COCO fallback.")
            self.cig_model = self.phone_model   # fallback (less accurate)

        self.phone_model.fuse()

    # ── Detection ─────────────────────────────────────────────────────────────
    def detect(self, frame):
        """
        Args:
            frame: BGR numpy array

        Returns:
            dict with keys:
                phone_detected      (bool)
                cigarette_detected  (bool)
                detections          (list of dicts with box, label, conf)
                frame               (annotated BGR frame)
                message             (str alert message)
        """
        phone_detected     = False
        cigarette_detected = False
        detections         = []
        alerts             = []

        # ── Run COCO model for phone ──────────────────────────────────────
        results = self.phone_model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        for box in results.boxes:
            cls_name = self.phone_model.names[int(box.cls)].lower()
            conf     = float(box.conf)
            xyxy     = box.xyxy[0].cpu().numpy().astype(int)

            if cls_name in PHONE_CLASSES:
                phone_detected = True
                detections.append({"label": "Phone", "conf": conf, "box": xyxy})
                self._draw_box(frame, xyxy, f"Phone {conf:.0%}", (0, 165, 255))
                alerts.append("PHONE USAGE ALERT!")

        # ── Run cigarette model ───────────────────────────────────────────
        if self.cig_model is not self.phone_model:
            cig_results = self.cig_model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
            for box in cig_results.boxes:
                cls_name = self.cig_model.names[int(box.cls)].lower()
                conf     = float(box.conf)
                xyxy     = box.xyxy[0].cpu().numpy().astype(int)

                if any(c in cls_name for c in CIGARETTE_CLASSES):
                    cigarette_detected = True
                    detections.append({"label": "Cigarette", "conf": conf, "box": xyxy})
                    self._draw_box(frame, xyxy, f"Cigarette {conf:.0%}", (128, 0, 128))
                    alerts.append("CIGARETTE ALERT!")

        message = " | ".join(alerts) if alerts else "No threats detected"

        return {
            "phone_detected":     phone_detected,
            "cigarette_detected": cigarette_detected,
            "detections":         detections,
            "frame":              frame,
            "message":            message
        }

    # ── Draw bounding box ─────────────────────────────────────────────────────
    @staticmethod
    def _draw_box(frame, xyxy, label, color):
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
