"""
Drowsiness Detection Module
Uses MediaPipe FaceMesh + Eye Aspect Ratio (EAR) algorithm
Compatible with mediapipe >= 0.10.x
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist

try:
    import mediapipe as mp
    OLD_SOLUTIONS = hasattr(mp, 'solutions')
except:
    OLD_SOLUTIONS = False

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
EAR_THRESHOLD  = 0.25
CONSEC_FRAMES  = 20


class DrowsinessDetector:
    def __init__(self):
        self.frame_counter = 0
        self.drowsy        = False
        self.ear_history   = []
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
            print("[DrowsinessDetector] Using new MediaPipe Tasks API")

    def _eye_aspect_ratio(self, landmarks, eye_indices, frame_w, frame_h):
        coords = np.array([(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in eye_indices])
        A = dist.euclidean(coords[1], coords[5])
        B = dist.euclidean(coords[2], coords[4])
        C = dist.euclidean(coords[0], coords[3])
        return (A + B) / (2.0 * C)

    def detect(self, frame):
        h, w   = frame.shape[:2]
        ear     = 0.0
        drowsy  = False
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
                    drowsy, message, frame = self._process_ear(ear, frame, w, h, lms)
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
                    drowsy, message, frame = self._process_ear(ear, frame, w, h, lms)
                else:
                    message = "No face detected"
        except Exception as e:
            message = f"Error: {str(e)[:40]}"

        self.ear_history.append(ear)
        if len(self.ear_history) > 30:
            self.ear_history.pop(0)
        self.drowsy = drowsy
        return {"drowsy": drowsy, "ear": round(ear, 3), "message": message, "frame": frame}

    def _process_ear(self, ear, frame, w, h, lms):
        drowsy  = False
        message = "Alert"
        if ear < EAR_THRESHOLD:
            self.frame_counter += 1
            if self.frame_counter >= CONSEC_FRAMES:
                drowsy  = True
                message = "DROWSINESS ALERT!"
        else:
            self.frame_counter = 0
        color = (0, 0, 255) if drowsy else (0, 255, 0)
        for idx in LEFT_EYE + RIGHT_EYE:
            lm = lms[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 2, color, -1)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if drowsy:
            cv2.putText(frame, message, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        return drowsy, message, frame

    def reset(self):
        self.frame_counter = 0
        self.drowsy        = False
        self.ear_history   = []
