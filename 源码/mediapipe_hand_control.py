import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import serial
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image as image_module
import logging
from collections import deque
import requests
import json
import threading

logging.getLogger('absl').setLevel(logging.ERROR)

DATA_FILE = 'gesture_data.npy'
LABEL_FILE = 'gesture_labels.npy'

SERVER_URL = "http://localhost:5000"

def upload_to_server(X, y):
    try:
        data = {"features": X.tolist(), "labels": y.tolist()}
        response = requests.post(f"{SERVER_URL}/train", json=data, timeout=30)
        if response.status_code == 200:
            print(f"Server: {response.json().get('message', 'Training complete')}")
            return True
        else:
            print(f"Server error: {response.status_code}")
    except Exception as e:
        print(f"Connection error: {e}")
    return False

def predict_server(features):
    try:
        data = {"features": features}
        response = requests.post(f"{SERVER_URL}/predict", json=data, timeout=5)
        if response.status_code == 200:
            return response.json().get('gesture')
    except:
        pass
    return None

def load_training_data():
    if os.path.exists(DATA_FILE) and os.path.exists(LABEL_FILE):
        X = np.load(DATA_FILE)
        y = np.load(LABEL_FILE)
        return X, y
    return None, None

def save_training_data(X, y):
    np.save(DATA_FILE, X)
    np.save(LABEL_FILE, y)
    print(f"Saved {len(y)} samples")

def train_model(X, y):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(X, y)
    return knn

def predict_gesture(model, features):
    if model is None:
        return None
    pred = model.predict([features])[0]
    return pred

def skin_detect(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return mask

try:
    for port in ['COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'COM3', 'COM4']:
        try:
            ser = serial.Serial(port, 115200, timeout=1)
            print(f"Connected to {port}")
            time.sleep(2)
            break
        except:
            continue
    else:
        ser = None
        print("Warning: No serial port found")
except:
    ser = None
    print("Warning: Serial port not available")

def move(s, a):
    if ser:
        try:
            ser.write(f"{s}{a}\r\n".encode())
        except:
            pass

model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print(f"Error: Model file not found {model_path}")
    print("Please download the model from:")
    print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    sys.exit(1)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    num_hands=1,
    min_hand_detection_confidence=0.05,
    min_hand_presence_confidence=0.05,
    min_tracking_confidence=0.05
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = frozenset([
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20),
    (5,9), (9,13), (13,17)
])

def get_finger_state(landmarks):
    states = []
    tips = [4, 8, 12, 16, 20]
    bases = [2, 5, 9, 13, 17]
    
    thumb = landmarks[tips[0]]
    thumb_base = landmarks[bases[0]]
    states.append(1 if thumb.x > thumb_base.x else 0)
    
    for i in range(1, 5):
        tip = landmarks[tips[i]]
        base = landmarks[bases[i]]
        states.append(1 if tip.y < base.y else 0)
    
    return states

def recognize_gesture(finger_states):
    if finger_states == [0, 0, 0, 0, 0]:
        return "Fist", 0
    elif finger_states == [1, 1, 1, 1, 1]:
        return "Palm", 1
    elif finger_states == [0, 1, 0, 0, 0]:
        return "Point", 2
    elif finger_states == [0, 1, 1, 0, 0]:
        return "Victory", 3
    elif finger_states == [1, 0, 0, 0, 0]:
        return "Thumbs Up", 4
    return f"{sum(finger_states)} Fingers", sum(finger_states)

GESTURE_NAMES = {
    0: "Fist",
    1: "Palm",
    2: "Point",
    3: "Victory",
    4: "Thumbs Up"
}

cap = cv2.VideoCapture(0)
for device_id in [0, 1]:
    cap = cv2.VideoCapture(device_id)
    if cap.isOpened():
        ret_test, frame_test = cap.read()
        if ret_test and frame_test is not None:
            print(f"Using camera device {device_id}")
            break
        cap.release()
else:
    print("ERROR: No camera available")
    sys.exit(1)

cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

X, y = load_training_data()
model = None
if X is not None and len(X) > 10:
    print(f"Training model with {len(y)} samples...")
    model = train_model(X, y)
    print("Model ready!")

collect_mode = False
collect_label = 0
collect_count = 0
mode = "TRACKING"
server_mode = False

last_time = time.time()
stable = 0
last_gesture = None
finger_states_buffer = deque(maxlen=5)
collected_X = []
collected_y = []

print("=== MediaPipe Hand Control ===")
print("Key Guide:")
print("  1-5: Collect gesture (1=Fist 2=Palm 3=Point 4=Victory 5=Thumbs Up)")
print("  T: Train model")
print("  C: Toggle collect mode")
print("  Q: Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    mask = skin_detect(frame)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
    mp_image = image_module.Image(image_format=image_module.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)
    
    vis_frame = frame.copy()
    
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key >= ord('1') and key <= ord('5'):
        collect_mode = True
        collect_label = key - ord('1')
        collect_count = 0
        mode = "COLLECT"
    elif key == ord('t') or key == ord('T'):
        if len(collected_y) > 10:
            X = np.array(collected_X)
            y = np.array(collected_y)
            save_training_data(X, y)
            model = train_model(X, y)
            mode = "ML"
            print(f"Model trained with {len(y)} samples!")
        else:
            print("Need more data. Collect more samples first.")
    elif key == ord('c') or key == ord('C'):
        collect_mode = not collect_mode
        mode = "COLLECT" if collect_mode else "TRACKING"
        print(f"Collect mode: {collect_mode}")
    elif key == ord('s') or key == ord('S'):
        server_mode = not server_mode
        mode = "SERVER" if server_mode else "TRACKING"
        print(f"Server mode: {server_mode}")
    elif key == ord('u') or key == ord('U'):
        if len(collected_X) > 10:
            X = np.array(collected_X)
            y = np.array(collected_y)
            if upload_to_server(X, y):
                print("Data uploaded to server!")
        else:
            print("Need more data to upload.")
    
    if results and results.hand_landmarks:
        landmarks = results.hand_landmarks[0]
        
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])
        
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(vis_frame, (x, y), 3, (0, 255, 0), -1)
        
        for edge in HAND_CONNECTIONS:
            p1 = landmarks[edge[0]]
            p2 = landmarks[edge[1]]
            cv2.line(vis_frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (255, 0, 255), 1)
        
        finger_states = get_finger_state(landmarks)
        
        finger_states_buffer.append(tuple(finger_states))
        
        avg_state = [0]*5
        for fs in finger_states_buffer:
            for i in range(5):
                avg_state[i] += fs[i]
        for i in range(5):
            avg_state[i] = 1 if avg_state[i] >= len(finger_states_buffer)/2 else 0
        
        if model is not None and mode == "ML":
            ml_gesture = predict_gesture(model, features)
            gesture = GESTURE_NAMES.get(ml_gesture, "Unknown")
        elif server_mode:
            server_gesture = predict_server(features)
            if server_gesture is not None:
                gesture = server_gesture
            else:
                gesture, _ = recognize_gesture(avg_state)
        else:
            gesture, _ = recognize_gesture(avg_state)
        
        if collect_mode:
            collected_X.append(features)
            collected_y.append(collect_label)
            collect_count += 1
        
        cx = int(landmarks[9].x * w)
        cy = int(landmarks[9].y * h)
        
        if last_gesture == gesture:
            stable = min(stable + 1, 3)
        else:
            stable = 0
        last_gesture = gesture
        
        cv2.circle(vis_frame, (cx, cy), 10, (0, 0, 255), -1)
        cv2.circle(vis_frame, (cx, cy), 14, (255, 255, 255), 2)
        
        cv2.putText(vis_frame, f"Mode:{mode} Samples:{len(collected_y)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(vis_frame, f"Gesture: {gesture}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(vis_frame, f"Thumb:{avg_state[0]} Index:{avg_state[1]} Middle:{avg_state[2]}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_frame, f"Ring:{avg_state[3]} Pinky:{avg_state[4]}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        s1 = 180 if avg_state[0] == 1 else 0
        s2 = 180 if avg_state[1] == 1 else 0
        s3 = 180 if avg_state[2] == 1 else 0
        s4 = 180 if avg_state[3] == 1 else 0
        s5 = 180 if avg_state[4] == 1 else 0
        
        cv2.putText(vis_frame, f"S1:{s1} S2:{s2} S3:{s3}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_frame, f"S4:{s4} S5:{s5}", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if stable >= 1 and time.time() - last_time > 0.03:
            move(1, s1)
            move(2, s2)
            move(3, s3)
            move(4, s4)
            move(5, s5)
            last_time = time.time()
    else:
        stable = 0
    
    cv2.imshow("MediaPipe Hand Control", vis_frame)

cap.release()
cv2.destroyAllWindows()

if len(collected_X) > 0:
    X = np.array(collected_X)
    y = np.array(collected_y)
    save_training_data(X, y)

if ser:
    ser.close()
print("Program ended!")