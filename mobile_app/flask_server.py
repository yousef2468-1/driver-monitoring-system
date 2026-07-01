import sys, os, socket
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flask import Flask
from flask_socketio import SocketIO, emit
import cv2, numpy as np, base64
from models.drowsiness_detector import DrowsinessDetector
from models.alert_manager       import AlertManager
from models.safety_score        import SafetyScore
from ultralytics import YOLO

app = Flask(__name__)
sio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', max_http_buffer_size=10e6)

print("[Server] Loading models...")
drowsy_det   = DrowsinessDetector()
alert_mgr    = AlertManager()
safety       = SafetyScore()
phone_model  = YOLO("yolov8n.pt"); phone_model.fuse()

# New smoking+vape model
smoke_model = None
smoke_path  = os.path.join(os.path.dirname(__file__), '..', 'models', 'weights', 'smoking_vape_cbam.pt')
if os.path.exists(smoke_path):
    smoke_model = YOLO(smoke_path)
    print(f"[Server] ✅ Smoking+Vape CBAM model loaded!")
else:
    print(f"[Server] ⚠️ No smoking model found!")

# MediaPipe for hand-to-ear
try:
    import mediapipe as mp
    OLD_SOLUTIONS = hasattr(mp, 'solutions')
except:
    OLD_SOLUTIONS = False

hand_detector = None
face_for_ear  = None

if OLD_SOLUTIONS:
    import mediapipe as mp
    hand_detector = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)
    face_for_ear  = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.4, min_tracking_confidence=0.4)
else:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import urllib.request
    hand_model_path = "/tmp/hand_landmarker.task"
    if not os.path.exists(hand_model_path):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            hand_model_path
        )
    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model_path),
            num_hands=1, min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5, min_tracking_confidence=0.5
        )
    )

phone_frames=0; gesture_frames=0; smoke_frames=0; CONFIRM_N=3
LEFT_EAR_IDX=[234,93,132,58,172]; RIGHT_EAR_IDX=[454,323,361,288,397]
print("[Server] ✅ All models loaded!")

def get_hand_pos(frame):
    h,w=frame.shape[:2]
    try:
        if OLD_SOLUTIONS:
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=hand_detector.process(rgb)
            if res.multi_hand_landmarks:
                lms=res.multi_hand_landmarks[0].landmark
                idx=np.array([lms[8].x*w,lms[8].y*h])
                pink=np.array([lms[20].x*w,lms[20].y*h])
                wrist=np.array([lms[0].x*w,lms[0].y*h])
                return (idx+pink)/2, wrist, idx
        else:
            import mediapipe as mp
            mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            res=hand_detector.detect(mp_img)
            if res.hand_landmarks:
                lms=res.hand_landmarks[0]
                idx=np.array([lms[8].x*w,lms[8].y*h])
                pink=np.array([lms[20].x*w,lms[20].y*h])
                wrist=np.array([lms[0].x*w,lms[0].y*h])
                return (idx+pink)/2, wrist, idx
    except: pass
    return None,None,None

def check_gesture(frame):
    h,w=frame.shape[:2]
    palm,wrist,idx_tip=get_hand_pos(frame)
    if palm is None: return False
    le=np.array([w*0.15,h*0.35]); re=np.array([w*0.85,h*0.35])
    thr=w*0.18
    dl=np.linalg.norm(palm-le); dr=np.linalg.norm(palm-re)
    if dl>thr and dr>thr: return False
    ear=le if dl<dr else re
    if wrist[1]<idx_tip[1]: return False
    if abs(palm[1]-ear[1])>h*0.20: return False
    cv2.circle(frame,(int(ear[0]),int(ear[1])),30,(0,165,255),3)
    cv2.circle(frame,(int(palm[0]),int(palm[1])),10,(0,165,255),-1)
    cv2.line(frame,(int(palm[0]),int(palm[1])),(int(ear[0]),int(ear[1])),(0,165,255),2)
    return True

@sio.on('connect')
def on_connect(): print("📱 Phone connected!")

@sio.on('disconnect')
def on_disconnect(): print("📱 Phone disconnected.")

@sio.on('frame')
def on_frame(data):
    global phone_frames, gesture_frames, smoke_frames
    try:
        img_data=data['image']
        if ',' in img_data: img_data=img_data.split(',')[1]
        nparr=np.frombuffer(base64.b64decode(img_data),np.uint8)
        frame=cv2.imdecode(nparr,cv2.IMREAD_COLOR)
        if frame is None: return

        # Drowsiness + Yawning
        d_res=drowsy_det.detect(frame); frame=d_res["frame"]

        # Phone: YOLO + Gesture
        phone_detected=False; phone_seen=False; gesture_seen=False
        try:
            results=phone_model(frame,conf=0.35,verbose=False)[0]
            for box in results.boxes:
                if phone_model.names[int(box.cls)].lower()=="cell phone":
                    phone_seen=True; phone_frames+=1
                    if phone_frames>=CONFIRM_N:
                        phone_detected=True
                        xyxy=box.xyxy[0].cpu().numpy().astype(int)
                        x1,y1,x2,y2=xyxy
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,165,255),2)
                        cv2.putText(frame,"PHONE",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,165,255),2)
        except: pass
        if not phone_seen: phone_frames=max(0,phone_frames-1)

        try:
            if check_gesture(frame):
                gesture_frames+=1
                if gesture_frames>=CONFIRM_N+1:
                    phone_detected=True
                    cv2.putText(frame,"HAND TO EAR!",(10,frame.shape[0]-40),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,165,255),3)
            else:
                gesture_frames=max(0,gesture_frames-1)
        except: pass

        # Smoking + Vape — hand-to-mouth gesture
        smoke_detected=False; smoke_seen=False
        try:
            h_,w_=frame.shape[:2]
            palm,wrist,idx_tip=get_hand_pos(frame)
            if palm is not None:
                mouth_y=h_*0.60
                in_center=w_*0.20<palm[0]<w_*0.80
                near_mouth=abs(palm[1]-mouth_y)<h_*0.25
                if in_center and near_mouth:
                    smoke_seen=True; smoke_frames+=1
                    if smoke_frames>=CONFIRM_N:
                        smoke_detected=True
                        cv2.putText(frame,"SMOKING!",(10,h_-40),
                                   cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,128),3)
        except: pass
        if not smoke_seen: smoke_frames=max(0,smoke_frames-1)

        # Alerts
        alerts=[]
        if d_res["drowsy"]:
            if alert_mgr.trigger("drowsiness","Drowsy!",True):
                safety.deduct("drowsiness"); alerts.append("😴 DROWSINESS!")
        if d_res["yawning"]:
            if alert_mgr.trigger("yawning","Yawning!",True):
                safety.deduct("yawning"); alerts.append("😮 YAWNING!")
        if phone_detected:
            if alert_mgr.trigger("phone","Phone!",True):
                safety.deduct("phone"); alerts.append("📱 PHONE!")
        if smoke_detected:
            if alert_mgr.trigger("cigarette","Smoking!",True):
                safety.deduct("cigarette"); alerts.append("🚬 SMOKING/VAPE!")

        s=safety.get_stats()
        _,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,65])
        img_b64="data:image/jpeg;base64,"+base64.b64encode(buf).decode()

        emit('result',{
            "drowsy":d_res["drowsy"],"yawning":d_res["yawning"],
            "phone":phone_detected,"cigarette":smoke_detected,
            "ear":d_res["ear"],"mar":d_res["mar"],"yawns":d_res["yawn_count"],
            "score":s["score"],"grade":s["grade"],"color":s["color"],
            "alerts":alerts,"image":img_b64,
        })
    except Exception as e: print(f"Error: {e}")

@sio.on('reset')
def on_reset():
    global phone_frames,gesture_frames,smoke_frames
    phone_frames=0; gesture_frames=0; smoke_frames=0
    alert_mgr.reset(); safety.reset()

if __name__=='__main__':
    try: ip=socket.gethostbyname(socket.gethostname())
    except: ip="10.0.0.213"
    print(f"\n  🚗 DMS — All 4 Detectors Active")
    print(f"  Drowsiness ✅ Yawning ✅ Phone ✅ Smoking/Vape ✅")
    print(f"  http://{ip}:5000\n")
    sio.run(app,host='0.0.0.0',port=5000,debug=False)
