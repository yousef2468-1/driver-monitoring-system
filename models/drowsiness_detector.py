import cv2
import numpy as np
from scipy.spatial import distance as dist
import os

try:
    import mediapipe as mp
    OLD_SOLUTIONS = hasattr(mp, 'solutions')
except:
    OLD_SOLUTIONS = False

LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
MOUTH_TOP  = 13
MOUTH_BOT  = 14
MOUTH_LEFT = 78
MOUTH_RIGHT= 308

EAR_THRESHOLD  = 0.23
MAR_THRESHOLD  = 0.55
CONSEC_FRAMES  = 5
YAWN_FRAMES    = 8

class DrowsinessDetector:
    def __init__(self):
        self.frame_counter = 0
        self.yawn_counter  = 0
        self.drowsy        = False
        self.yawning       = False
        self.ear_history   = []
        self.mar_history   = []
        self.yawn_count    = 0
        # EAR buffer for smoothing
        self.ear_buffer    = []
        self.BUFFER_SIZE   = 5
        self._init_mediapipe()

    def _init_mediapipe(self):
        if OLD_SOLUTIONS:
            import mediapipe as mp
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.mode = "old"
        else:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import urllib.request

            model_path = "/tmp/face_landmarker.task"
            if not os.path.exists(model_path):
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                    model_path
                )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=model_path),
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1,
                    min_face_detection_confidence=0.3,
                    min_face_presence_confidence=0.3,
                    min_tracking_confidence=0.3
                )
            )
            self.mode = "new"

    def _ear(self, lms, idx, w, h):
        c = np.array([(lms[i].x*w, lms[i].y*h) for i in idx])
        A = dist.euclidean(c[1],c[5])
        B = dist.euclidean(c[2],c[4])
        C = dist.euclidean(c[0],c[3])
        return (A+B)/(2.0*C)

    def _mar(self, lms, w, h):
        t = np.array([lms[MOUTH_TOP].x*w,   lms[MOUTH_TOP].y*h])
        b = np.array([lms[MOUTH_BOT].x*w,   lms[MOUTH_BOT].y*h])
        l = np.array([lms[MOUTH_LEFT].x*w,  lms[MOUTH_LEFT].y*h])
        r = np.array([lms[MOUTH_RIGHT].x*w, lms[MOUTH_RIGHT].y*h])
        return dist.euclidean(t,b)/(dist.euclidean(l,r)+1e-6)

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
                    ear = (self._ear(lms,LEFT_EYE,w,h)+self._ear(lms,RIGHT_EYE,w,h))/2
                    mar = self._mar(lms,w,h)
                    drowsy, yawning, message, frame = self._process(ear,mar,frame,w,h,lms)
                else:
                    # Use last known EAR if face lost
                    if self.ear_buffer:
                        ear = np.mean(self.ear_buffer)
                    message = "Tracking..."
            else:
                import mediapipe as mp
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = self.face_landmarker.detect(mp_img)
                if result.face_landmarks:
                    lms = result.face_landmarks[0]
                    ear = (self._ear(lms,LEFT_EYE,w,h)+self._ear(lms,RIGHT_EYE,w,h))/2
                    mar = self._mar(lms,w,h)
                    drowsy, yawning, message, frame = self._process(ear,mar,frame,w,h,lms)
                else:
                    if self.ear_buffer:
                        ear = np.mean(self.ear_buffer)
                    message = "Tracking..."
        except Exception as e:
            message = f"Error: {str(e)[:30]}"

        # Update buffers
        if ear > 0:
            self.ear_buffer.append(ear)
            if len(self.ear_buffer) > self.BUFFER_SIZE:
                self.ear_buffer.pop(0)

        self.ear_history.append(ear)
        self.mar_history.append(mar)
        if len(self.ear_history) > 60: self.ear_history.pop(0)
        if len(self.mar_history) > 60: self.mar_history.pop(0)

        self.drowsy  = drowsy
        self.yawning = yawning

        return {"drowsy":drowsy,"yawning":yawning,"yawn_count":self.yawn_count,
                "ear":round(ear,3),"mar":round(mar,3),"message":message,"frame":frame}

    def _process(self, ear, mar, frame, w, h, lms):
        drowsy  = False
        yawning = False
        alerts  = []

        # Smooth EAR using buffer average
        self.ear_buffer.append(ear)
        if len(self.ear_buffer) > self.BUFFER_SIZE:
            self.ear_buffer.pop(0)
        smooth_ear = np.mean(self.ear_buffer) if self.ear_buffer else ear

        # Drowsiness check with smoothed EAR
        if smooth_ear < EAR_THRESHOLD:
            self.frame_counter += 1
            if self.frame_counter >= CONSEC_FRAMES:
                drowsy = True
                alerts.append("DROWSINESS!")
        else:
            self.frame_counter = max(0, self.frame_counter - 1)

        # Yawning check
        if mar > MAR_THRESHOLD:
            self.yawn_counter += 1
            if self.yawn_counter >= YAWN_FRAMES:
                yawning = True
                alerts.append("YAWNING!")
        else:
            if self.yawn_counter >= YAWN_FRAMES:
                self.yawn_count += 1
            self.yawn_counter = 0

        # Draw
        eye_c = (0,0,255) if drowsy else (0,255,0)
        mth_c = (0,165,255) if yawning else (0,255,255)

        for idx in LEFT_EYE+RIGHT_EYE:
            lm = lms[idx]
            cv2.circle(frame,(int(lm.x*w),int(lm.y*h)),2,eye_c,-1)
        for idx in [MOUTH_TOP,MOUTH_BOT,MOUTH_LEFT,MOUTH_RIGHT]:
            lm = lms[idx]
            cv2.circle(frame,(int(lm.x*w),int(lm.y*h)),3,mth_c,-1)

        cv2.putText(frame,f"EAR:{smooth_ear:.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,eye_c,2)
        cv2.putText(frame,f"MAR:{mar:.2f}",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.6,mth_c,2)
        cv2.putText(frame,f"Yawns:{self.yawn_count}",(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        for i, a in enumerate(alerts):
            cv2.putText(frame,a,(10,110+i*35),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),3)

        return drowsy, yawning, " | ".join(alerts) if alerts else "Alert", frame

    def reset(self):
        self.frame_counter = 0
        self.yawn_counter  = 0
        self.drowsy        = False
        self.yawning       = False
        self.yawn_count    = 0
        self.ear_history   = []
        self.mar_history   = []
        self.ear_buffer    = []
