import cv2
import numpy as np
from ultralytics import YOLO
import os

try:
    import mediapipe as mp
    OLD_SOLUTIONS = hasattr(mp, 'solutions')
except:
    OLD_SOLUTIONS = False

CONF_THRESHOLD = 0.30
SMOKE_CLASSES  = {"cigarette", "smoking", "smoke", "vape"}

# Ear landmark indices
LEFT_EAR_IDX  = [234, 93, 132, 58, 172]
RIGHT_EAR_IDX = [454, 323, 361, 288, 397]

# Hand landmarks
WRIST      = 0
INDEX_TIP  = 8
PINKY_TIP  = 20
THUMB_TIP  = 4

class ObjectDetector:
    def __init__(self, cigarette_model_path=None):
        print("[ObjectDetector] Loading YOLOv8n...")
        self.yolo = YOLO("yolov8n.pt")
        self.yolo.fuse()

        # Try multiple paths for cigarette model
        self.cig_model = None
        possible_paths = [
            cigarette_model_path,
            os.path.join(os.path.dirname(__file__), "weights", "cigarette_model.pt"),
            os.path.join(os.path.dirname(__file__), "weights", "cigarette.pt"),
            os.path.join(os.path.dirname(__file__), "weights", "best.pt"),
        ]
        for path in possible_paths:
            if path and os.path.exists(path):
                print(f"[ObjectDetector] Cigarette model found: {path}")
                self.cig_model = YOLO(path)
                break
        if not self.cig_model:
            print("[ObjectDetector] No cigarette model found in weights/")

        # Phone gesture state
        self.phone_frames = 0
        self.cig_frames   = 0
        self.CONFIRM_N    = 4

        self._init_mediapipe()

    def _init_mediapipe(self):
        if OLD_SOLUTIONS:
            import mediapipe as mp
            self.mp_hands  = mp.solutions.hands
            self.hands     = self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_face   = mp.solutions.face_mesh
            self.face_mesh = self.mp_face.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mode = "old"
        else:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import urllib.request

            hand_model = "/tmp/hand_landmarker.task"
            if not os.path.exists(hand_model):
                print("[ObjectDetector] Downloading hand model...")
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    hand_model
                )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=hand_model),
                    num_hands=1,
                    min_hand_detection_confidence=0.7,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            )
            face_model = "/tmp/face_landmarker.task"
            if not os.path.exists(face_model):
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                    face_model
                )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=face_model),
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            )
            self.mode = "new"

    def _get_ear_positions(self, frame):
        h, w = frame.shape[:2]
        try:
            if self.mode == "old":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    lms = res.multi_face_landmarks[0].landmark
                    le = np.mean([(lms[i].x*w, lms[i].y*h) for i in LEFT_EAR_IDX], axis=0)
                    re = np.mean([(lms[i].x*w, lms[i].y*h) for i in RIGHT_EAR_IDX], axis=0)
                    return le, re
            else:
                import mediapipe as mp
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = self.face_landmarker.detect(mp_img)
                if res.face_landmarks:
                    lms = res.face_landmarks[0]
                    le = np.mean([(lms[i].x*w, lms[i].y*h) for i in LEFT_EAR_IDX], axis=0)
                    re = np.mean([(lms[i].x*w, lms[i].y*h) for i in RIGHT_EAR_IDX], axis=0)
                    return le, re
        except:
            pass
        return None, None

    def _get_hand_info(self, frame):
        h, w = frame.shape[:2]
        try:
            if self.mode == "old":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)
                if res.multi_hand_landmarks:
                    lms = res.multi_hand_landmarks[0].landmark
                    return {
                        "wrist":     np.array([lms[WRIST].x*w,     lms[WRIST].y*h]),
                        "index_tip": np.array([lms[INDEX_TIP].x*w, lms[INDEX_TIP].y*h]),
                        "pinky_tip": np.array([lms[PINKY_TIP].x*w, lms[PINKY_TIP].y*h]),
                        "thumb_tip": np.array([lms[THUMB_TIP].x*w, lms[THUMB_TIP].y*h]),
                    }
            else:
                import mediapipe as mp
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = self.hand_landmarker.detect(mp_img)
                if res.hand_landmarks:
                    lms = res.hand_landmarks[0]
                    return {
                        "wrist":     np.array([lms[WRIST].x*w,     lms[WRIST].y*h]),
                        "index_tip": np.array([lms[INDEX_TIP].x*w, lms[INDEX_TIP].y*h]),
                        "pinky_tip": np.array([lms[PINKY_TIP].x*w, lms[PINKY_TIP].y*h]),
                        "thumb_tip": np.array([lms[THUMB_TIP].x*w, lms[THUMB_TIP].y*h]),
                    }
        except:
            pass
        return None

    def _check_phone_gesture(self, hand, left_ear, right_ear, frame_w, frame_h):
        """
        Phone-to-ear: hand near ear + wrist below fingertips + at ear height
        Returns (is_phone, ear_position)
        """
        if hand is None or left_ear is None:
            return False, None

        wrist     = hand["wrist"]
        index_tip = hand["index_tip"]
        pinky_tip = hand["pinky_tip"]
        palm      = (index_tip + pinky_tip) / 2

        # 1. Near ear
        threshold = frame_w * 0.16
        dl = np.linalg.norm(palm - left_ear)
        dr = np.linalg.norm(palm - right_ear)
        if dl > threshold and dr > threshold:
            return False, None
        closest_ear = left_ear if dl < dr else right_ear

        # 2. Hand upright (wrist BELOW fingertips in image = y is larger)
        if wrist[1] < index_tip[1]:
            return False, None

        # 3. Palm at ear height (within 15% of frame height)
        if abs(palm[1] - closest_ear[1]) > frame_h * 0.15:
            return False, None

        # 4. Fingers closed (like holding phone) — index and pinky not too spread
        spread = np.linalg.norm(index_tip - pinky_tip)
        if spread > frame_w * 0.20:
            return False, None

        return True, closest_ear

    def detect(self, frame):
        h, w = frame.shape[:2]
        phone_detected     = False
        cigarette_detected = False
        detections         = []
        alerts             = []
        phone_seen         = False
        cig_seen           = False

        # ── Phone: hand-to-ear gesture ────────────────────────────────────
        try:
            left_ear, right_ear = self._get_ear_positions(frame)
            hand = self._get_hand_info(frame)
            is_phone, ear_pos = self._check_phone_gesture(
                hand, left_ear, right_ear, w, h)

            if is_phone:
                phone_seen = True
                self.phone_frames += 1
                if self.phone_frames >= self.CONFIRM_N:
                    phone_detected = True
                    alerts.append("PHONE TO EAR!")
                    palm = (hand["index_tip"] + hand["pinky_tip"]) / 2
                    cv2.circle(frame, (int(ear_pos[0]), int(ear_pos[1])), 35, (0,165,255), 3)
                    cv2.circle(frame, (int(palm[0]), int(palm[1])), 12, (0,165,255), -1)
                    cv2.line(frame, (int(palm[0]),int(palm[1])),
                                    (int(ear_pos[0]),int(ear_pos[1])), (0,165,255), 2)
                    cv2.putText(frame, f"PHONE TO EAR!", (10, h-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,165,255), 3)
                else:
                    cv2.putText(frame, f"Checking... ({self.phone_frames}/{self.CONFIRM_N})",
                                (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)
        except Exception as e:
            pass

        if not phone_seen:
            self.phone_frames = max(0, self.phone_frames - 1)

        # ── Cigarette / Vape ──────────────────────────────────────────────
        try:
            if self.cig_model:
                cig_res = self.cig_model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
                for box in cig_res.boxes:
                    cls_name = self.cig_model.names[int(box.cls)].lower()
                    conf     = float(box.conf)
                    xyxy     = box.xyxy[0].cpu().numpy().astype(int)
                    if any(c in cls_name for c in SMOKE_CLASSES):
                        cig_seen = True
                        self.cig_frames += 1
                        if self.cig_frames >= self.CONFIRM_N:
                            cigarette_detected = True
                            label = "Vape" if "vape" in cls_name else "Cigarette"
                            detections.append({"label": label, "conf": conf, "box": xyxy})
                            self._draw_box(frame, xyxy, f"{label} {conf:.0%}", (128,0,128))
                            alerts.append(f"{label.upper()} DETECTED!")
            else:
                cv2.putText(frame, "No smoke model loaded", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        except Exception as e:
            pass

        if not cig_seen:
            self.cig_frames = max(0, self.cig_frames - 1)

        return {
            "phone_detected":     phone_detected,
            "cigarette_detected": cigarette_detected,
            "detections":         detections,
            "frame":              frame,
            "message":            " | ".join(alerts) if alerts else "No threats detected",
        }

    @staticmethod
    def _draw_box(frame, xyxy, label, color):
        x1,y1,x2,y2 = xyxy
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.6,1)
        cv2.rectangle(frame,(x1,y1-th-8),(x1+tw+4,y1),color,-1)
        cv2.putText(frame,label,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
