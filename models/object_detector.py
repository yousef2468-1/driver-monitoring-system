import cv2, numpy as np, os
from ultralytics import YOLO

CONF_THRESHOLD = 0.30

class ObjectDetector:
    def __init__(self, cigarette_model_path=None):
        self.yolo = YOLO("yolov8n.pt")
        self.yolo.fuse()
        self.cig_model   = None
        self.phone_model = None
        base = os.path.dirname(__file__)
        for path in [
            os.path.join(base,"weights","smoking_cbam.pt"),
            os.path.join(base,"weights","cigarette_model.pt"),
        ]:
            if os.path.exists(path):
                self.cig_model = YOLO(path)
                print(f"[OD] Smoking: {path}")
                break
        phone_path = os.path.join(base,"weights","phone_cbam.pt")
        if os.path.exists(phone_path):
            self.phone_model = YOLO(phone_path)
            print(f"[OD] Phone: {phone_path}")
        self.phone_frames = 0
        self.cig_frames   = 0
        self.CONFIRM_N    = 2

    def detect(self, frame):
        phone_detected = False
        cigarette_detected = False
        detections = []
        alerts = []
        phone_seen = False
        cig_seen   = False
        try:
            if self.phone_model:
                result = self.phone_model(frame, verbose=False)[0]
                if result.probs is not None:
                    cls_id   = result.probs.top1
                    cls_name = self.phone_model.names[cls_id]
                    conf     = float(result.probs.top1conf)
                    if cls_name == 'phone_call' and conf > 0.6:
                        phone_seen = True
                        self.phone_frames += 1
                        if self.phone_frames >= self.CONFIRM_N:
                            phone_detected = True
                            alerts.append("PHONE DETECTED!")
                            cv2.putText(frame, f"PHONE! {conf:.0%}",
                                       (10, frame.shape[0]-40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,165,255), 3)
            else:
                results = self.yolo(frame, conf=CONF_THRESHOLD, verbose=False)[0]
                for box in results.boxes:
                    if self.yolo.names[int(box.cls)].lower() == "cell phone":
                        phone_seen = True
                        self.phone_frames += 1
                        if self.phone_frames >= self.CONFIRM_N:
                            phone_detected = True
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            self._draw_box(frame, xyxy, f"Phone {float(box.conf):.0%}", (0,165,255))
                            alerts.append("PHONE DETECTED!")
        except: pass

        if not phone_seen:
            self.phone_frames = max(0, self.phone_frames - 1)

        try:
            if self.cig_model:
                cig_res = self.cig_model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
                for box in cig_res.boxes:
                    cls_name = self.cig_model.names[int(box.cls)].lower()
                    conf     = float(box.conf)
                    xyxy     = box.xyxy[0].cpu().numpy().astype(int)
                    if any(c in cls_name for c in ["cigarette","smoke","vape"]):
                        cig_seen = True
                        self.cig_frames += 1
                        if self.cig_frames >= self.CONFIRM_N:
                            cigarette_detected = True
                            self._draw_box(frame, xyxy, f"Smoke {conf:.0%}", (128,0,128))
                            alerts.append("SMOKING DETECTED!")
        except: pass

        if not cig_seen:
            self.cig_frames = max(0, self.cig_frames - 1)

        return {"phone_detected": phone_detected, "cigarette_detected": cigarette_detected,
                "detections": detections, "frame": frame,
                "message": " | ".join(alerts) if alerts else "No threats"}

    @staticmethod
    def _draw_box(frame, xyxy, label, color):
        x1,y1,x2,y2 = xyxy
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.6,1)
        cv2.rectangle(frame,(x1,y1-th-8),(x1+tw+4,y1),color,-1)
        cv2.putText(frame,label,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
