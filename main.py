import cv2
import torch
import numpy as np
import threading
import time
import requests
import mediapipe as mp
from flask import Flask, Response

SIGNALWIRE_PROJECT = "your-project-id"
SIGNALWIRE_TOKEN = "your-api-token"
SIGNALWIRE_SPACE = "your-space.signalwire.com"
FROM_NUMBER = "+1YOUR_SIGNALWIRE_NUMBER"
TO_NUMBER = "+911"  

CAMERA_LOCATION = "your-cctv-area-name"
TWIML_PORT = 5000
TWIML_URL = f"http://localhost:{TWIML_PORT}/twiml"

last_called_times = {
    "fire": 0,
    "crime": 0,
    "fall": 0,
    "accident": 0
}
CALL_COOLDOWN = 60  
app = Flask(__name__)

latest_message = "No emergency."

@app.route("/twiml")
def twiml():
    global latest_message
    return Response(f"""
    <?xml version="1.0" encoding="UTF-8"?>
    <Response><Say voice="man">{latest_message}</Say></Response>
    """, mimetype='text/xml')

def run_flask():
    app.run(host="0.0.0.0", port=TWIML_PORT)

print("[🧠 Loading YOLOv5 model...]")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False)
print("[✅ YOLOv5 model loaded]")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def make_call(emergency_type, message):
    global latest_message, last_called_times
    now = time.time()
    if now - last_called_times.get(emergency_type, 0) < CALL_COOLDOWN:
        print(f"[⏳ Cooldown active for {emergency_type}, skipping call]")
        return
    last_called_times[emergency_type] = now

    latest_message = message
    payload = {
        "From": FROM_NUMBER,
        "To": TO_NUMBER,
        "Url": TWIML_URL
    }
    try:
        url = f"https://{SIGNALWIRE_SPACE}/api/laml/2010-04-01/Accounts/{SIGNALWIRE_PROJECT}/Calls.json"
        response = requests.post(url, auth=(SIGNALWIRE_PROJECT, SIGNALWIRE_TOKEN), data=payload)
        if response.status_code == 201:
            print(f"[📞 CALL PLACED for {emergency_type}]")
        else:
            print(f"[❌ CALL ERROR {response.status_code}]: {response.text}")
    except Exception as e:
        print(f"[❗ CALL EXCEPTION]: {e}")

def generate_message(emergency_type, description):
    return (
        f"This is an automated emergency alert from AI CCTV monitoring system at {CAMERA_LOCATION}. "
        f"A case of {emergency_type} has been detected: {description}. Please dispatch emergency services immediately."
    )

def detect_fire(frame):
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 50, 50])
        upper = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        fire_pixels = cv2.countNonZero(mask)
        if fire_pixels > 1000:
            return True
        return False
    except Exception as e:
        print(f"[❗ FIRE DETECTION ERROR]: {e}")
        return False

def detect_weapons(results):
    try:
        weapons = ["knife", "gun"]
        found = []
        for *box, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            if label in weapons:
                found.append(label)
        return found
    except Exception as e:
        print(f"[❗ WEAPON DETECTION ERROR]: {e}")
        return []

fall_threshold = 0.5
fall_cooldown_seconds = 15
last_fall_time = 0

def detect_fall(frame):
    global last_fall_time
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

            torso_height = abs(shoulder.y - hip.y)
            torso_width = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x)

            ratio = torso_height / torso_width if torso_width != 0 else 0

            if ratio < fall_threshold:
                now = time.time()
                if now - last_fall_time > fall_cooldown_seconds:
                    last_fall_time = now
                    return True
        return False
    except Exception as e:
        print(f"[❗ FALL DETECTION ERROR]: {e}")
        return False

last_vehicle_boxes = []
vehicle_labels = ["car", "truck", "bus", "motorbike"]

def detect_accident(results):
    global last_vehicle_boxes
    try:
        current_boxes = []
        accident_detected = False

        for *box, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            if label in vehicle_labels:
                x1, y1, x2, y2 = map(int, box)
                current_boxes.append((x1, y1, x2, y2))

        for (x1, y1, x2, y2) in current_boxes:
            for (lx1, ly1, lx2, ly2) in last_vehicle_boxes:
                dx = min(x2, lx2) - max(x1, lx1)
                dy = min(y2, ly2) - max(y1, ly1)
                if dx > 0 and dy > 0:
                    overlap_area = dx * dy
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (lx2 - lx1) * (ly2 - ly1)
                    overlap_ratio = overlap_area / min(area1, area2)
                    if overlap_ratio > 0.3:
                        accident_detected = True
                        break
            if accident_detected:
                break

        last_vehicle_boxes = current_boxes
        return accident_detected
    except Exception as e:
        print(f"[❗ ACCIDENT DETECTION ERROR]: {e}")
        return False

def process_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[❌ ERROR: Cannot open webcam]")
        return

    print("[▶️ Starting video capture. Press ESC to exit.]")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[⚠️ VIDEO STREAM ENDED]")
            break

        if detect_fire(frame):
            msg = generate_message("fire", "flames or smoke detected")
            threading.Thread(target=make_call, args=("fire", msg)).start()
            time.sleep(10)

        results = model(frame)

        weapons = detect_weapons(results)
        if weapons:
            msg = generate_message("crime", f"weapons detected: {', '.join(weapons)}")
            threading.Thread(target=make_call, args=("crime", msg)).start()
            time.sleep(10)
        if detect_accident(results):
            msg = generate_message("accident", "vehicle collision detected")
            threading.Thread(target=make_call, args=("accident", msg)).start()
            time.sleep(10)
        if detect_fall(frame):
            msg = generate_message("fall", "person has fallen or fainted")
            threading.Thread(target=make_call, args=("fall", msg)).start()
            time.sleep(10)
        annotated_frame = np.squeeze(results.render())
        cv2.imshow("AI CCTV Emergency Detection", annotated_frame)

        if cv2.waitKey(1) == 27:  
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    print("[✅ AI CCTV MONITORING STARTED]")
    process_video()
