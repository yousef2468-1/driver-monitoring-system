"""
Drowsiness Detection Module
Uses MediaPipe FaceMesh + Eye Aspect Ratio (EAR) algorithm
No training needed - works out of the box
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist


# ── MediaPipe landmark indices for eyes ──────────────────────────────────────
# Left eye landmarks
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye landmarks
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ── Thresholds ────────────────────────────────────────────────────────────────
EAR_THRESHOLD      = 0.25   # below this → eyes considered closed
CONSEC_FRAMES      = 20     # frames eye must be closed to trigger alert
YAWN_THRESHOLD     = 0.6    # mouth open ratio threshold


class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.frame_counter = 0
        self.drowsy        = False
        self.ear_history   = []

    # ── Eye Aspect Ratio ─────────────────────────────────────────────────────
    def _eye_aspect_ratio(self, landmarks, eye_indices, frame_w, frame_h):
        coords = []
        for idx in eye_indices:
            lm = landmarks[idx]
            coords.append((lm.x * frame_w, lm.y * frame_h))
        coords = np.array(coords)

        # vertical distances
        A = dist.euclidean(coords[1], coords[5])
        B = dist.euclidean(coords[2], coords[4])
        # horizontal distance
        C = dist.euclidean(coords[0], coords[3])

        ear = (A + B) / (2.0 * C)
        return ear

    # ── Process one frame ────────────────────────────────────────────────────
    def detect(self, frame):
        """
        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            dict with keys:
                drowsy   (bool)
                ear      (float)
                message  (str)
                frame    (annotated BGR frame)
        """
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        ear     = 0.0
        drowsy  = False
        message = "Alert"

        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark

            left_ear  = self._eye_aspect_ratio(lms, LEFT_EYE,  w, h)
            right_ear = self._eye_aspect_ratio(lms, RIGHT_EYE, w, h)
            ear       = (left_ear + right_ear) / 2.0

            self.ear_history.append(ear)
            if len(self.ear_history) > 30:
                self.ear_history.pop(0)

            if ear < EAR_THRESHOLD:
                self.frame_counter += 1
                if self.frame_counter >= CONSEC_FRAMES:
                    drowsy  = True
                    message = "DROWSINESS ALERT!"
            else:
                self.frame_counter = 0

            # ── Draw eye landmarks ────────────────────────────────────────
            color = (0, 0, 255) if drowsy else (0, 255, 0)
            for idx in LEFT_EYE + RIGHT_EYE:
                lm = lms[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, color, -1)

            # ── HUD ───────────────────────────────────────────────────────
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if drowsy:
                cv2.putText(frame, message, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            message = "No face detected"

        self.drowsy = drowsy
        return {
            "drowsy":  drowsy,
            "ear":     round(ear, 3),
            "message": message,
            "frame":   frame
        }

    def reset(self):
        self.frame_counter = 0
        self.drowsy        = False
        self.ear_history   = []
