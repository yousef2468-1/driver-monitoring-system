"""
Drowsiness + Yawning Detection Module v2
- Eye Aspect Ratio (EAR) for drowsiness
- Mouth Aspect Ratio (MAR) for yawning
- Compatible with mediapipe >= 0.10.x
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist

try:
    import mediapipe as mp
    OLD_SOLUTIONS = hasattr(mp, 'solutions')
except:
    OLD_SOLUTIONS = False

# ── Eye landmarks ─────────────────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ── Mouth landmarks for yawning ───────────────────────────────────────────────
# Top lip, bottom lip, left corner, right corner
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 78
MOUTH_RIGHT  = 308
MOUTH_TOP2   = 312
MOUTH_BOT2   = 317

# ── Thresholds ────────────────────────────────────────────────────────────────
EAR_THRESHOLD   = 0.25
MAR_THRESHOLD   = 0.6    # Mouth open ratio for yawning
CONSEC_FRAMES   = 20     # ~0.67s at 30fps for drowsiness
YAWN_FRAMES     = 15     # frames mouth must be open to count as yawn


class DrowsinessDetector:
    def __init__(self):
        self.frame_counter  = 0
        self.yawn_counter   = 0
        self.drowsy         = False
        self.yawning        = False
        self.ear_history    = []
        self.mar_history    = []
        self.yawn_count     = 0    # total yawns this session
        self._init_detector()

    def _init_detector(self):
        if OLD_SOLUTIONS:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mode = "old"
        else:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import urllib.request, os

            model_path = "/tmp/face_landmarker.task"
            if not os.path.exists(model_path):
                print("[DrowsinessDetector] Downloading face landmarker model...")
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                    model_path
                )
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            self.mode = "new"

    def _eye_aspect_ratio(self, landmarks, eye_indices, w, h):
        coords = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices])
        A = dist.euclidean(coords[1], coords[5])
        B = dist.euclidean(coords[2], coords[4])
        C = dist.euclidean(coords[0], coords[3])
        return (A + B) / (2.0 * C)

    def _mouth_aspect_ratio(self, landmarks, w, h):
        top    = np.array([landmarks[MOUTH_TOP].x * w,    landmarks[MOUTH_TOP].y * h])
        bottom = np.array([landmarks[MOUTH_BOTTOM].x * w, landmarks[MOUTH_BOTTOM].y * h])
        left   = np.array([landmarks[MOUTH_LEFT].x * w,   landmarks[MOUTH_LEFT].y * h])
        right  = np.array([landmarks[MOUTH_RIGHT].x * w,  landmarks[MOUTH_RIGHT].y * h])
        vertical   = dist.euclidean(top, bottom)
        horizontal = dist.euclidean(left, right)
        return vertical / (horizontal + 1e-6)

    def detect(self, frame):
        h, w    = frame.shape[:2]
        ear     = 0.0
        mar     = 0.0
        drowsy  = False
        yawning = False
        message = "Alert"

        try:
            if self.mode == "old":
                import mediapipe as mp
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.face_mesh.process(rgb)
                if result.multi_face_landmarks:
                    lms = result.multi_face_landmarks[0].landmark
                    ear = (self._eye_aspect_ratio(lms, LEFT_EYE, w, h) +
                           self._eye_aspect_ratio(lms, RIGHT_EYE, w, h)) / 2.0
                    mar = self._mouth_aspect_ratio(lms, w, h)
                    drowsy, yawning, message, frame = self._process(ear, mar, frame, w, h, lms)
                else:
                    message = "No face detected"
            else:
                import mediapipe as mp
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result    = self.face_landmarker.detect(mp_image)
                if result.face_landmarks:
                    lms = result.face_landmarks[0]
                    ear = (self._eye_aspect_ratio(lms, LEFT_EYE, w, h) +
                           self._eye_aspect_ratio(lms, RIGHT_EYE, w, h)) / 2.0
                    mar = self._mouth_aspect_ratio(lms, w, h)
                    drowsy, yawning, message, frame = self._process(ear, mar, frame, w, h, lms)
                else:
                    message = "No face detected"
        except Exception as e:
            message = f"Error: {str(e)[:40]}"

        self.ear_history.append(ear)
        self.mar_history.append(mar)
        if len(self.ear_history) > 60: self.ear_history.pop(0)
        if len(self.mar_history) > 60: self.mar_history.pop(0)

        self.drowsy  = drowsy
        self.yawning = yawning

        return {
            "drowsy":      drowsy,
            "yawning":     yawning,
            "yawn_count":  self.yawn_count,
            "ear":         round(ear, 3),
            "mar":         round(mar, 3),
            "message":     message,
            "frame":       frame
        }

    def _process(self, ear, mar, frame, w, h, lms):
        drowsy  = False
        yawning = False
        alerts  = []

        # ── Drowsiness (EAR) ──────────────────────────────────────────────
        if ear < EAR_THRESHOLD:
            self.frame_counter += 1
            if self.frame_counter >= CONSEC_FRAMES:
                drowsy = True
                alerts.append("DROWSINESS ALERT!")
        else:
            self.frame_counter = 0

        # ── Yawning (MAR) ─────────────────────────────────────────────────
        if mar > MAR_THRESHOLD:
            self.yawn_counter += 1
            if self.yawn_counter >= YAWN_FRAMES:
                yawning = True
                alerts.append("YAWNING DETECTED!")
        else:
            if self.yawn_counter >= YAWN_FRAMES:
                self.yawn_count += 1   # completed yawn
            self.yawn_counter = 0

        # ── Draw eye landmarks ────────────────────────────────────────────
        eye_color = (0, 0, 255) if drowsy else (0, 255, 0)
        for idx in LEFT_EYE + RIGHT_EYE:
            lm = lms[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 2, eye_color, -1)

        # ── Draw mouth landmarks ──────────────────────────────────────────
        mouth_color = (0, 165, 255) if yawning else (0, 255, 255)
        for idx in [MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT]:
            lm = lms[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, mouth_color, -1)

        # ── HUD ───────────────────────────────────────────────────────────
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_color, 2)
        cv2.putText(frame, f"Yawns: {self.yawn_count}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for i, alert in enumerate(alerts):
            cv2.putText(frame, alert, (10, 110 + i*35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        message = " | ".join(alerts) if alerts else "Alert"
        return drowsy, yawning, message, frame

    def reset(self):
        self.frame_counter = 0
        self.yawn_counter  = 0
        self.drowsy        = False
        self.yawning       = False
        self.yawn_count    = 0
        self.ear_history   = []
        self.mar_history   = []
