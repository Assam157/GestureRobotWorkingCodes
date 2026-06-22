 
# =========================================================
# PREDICTIVE CLOUD ROBOTICS
# LOCAL TEST + OPTIONAL WEBSOCKET
# =========================================================


import cv2
import asyncio
import websockets
import json
import time
import math
import numpy as np
import mediapipe as mp
import nest_asyncio

nest_asyncio.apply()

# =========================================================
# CONFIG
# =========================================================

HOST = "0.0.0.0"
PORT = 8765

WAIT_FOR_CLIENT_SECONDS = 10

PREDICTION_FACTOR = 5
INTERPOLATION_SPEED = 0.15

# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================================================
# ACTIVE CLIENTS
# =========================================================

connected_clients = set()

# =========================================================
#  FILTERS
# =========================================================
import random

class ParticleFilter:

    def __init__(self,
                 particles=100):

        self.particles = [0] * particles

    def update(self, measurement):

        self.particles = [

            measurement +

            random.uniform(-0.05,0.05)

            for _ in self.particles
        ]

        return (
            sum(self.particles) /
            len(self.particles)
        )

    def predict_future(self):

        return (
            sum(self.particles) /
            len(self.particles)
        )

kf_x = ParticleFilter()
kf_y = ParticleFilter()
kf_z = ParticleFilter()
# =========================================================
# SERVO STATE
# =========================================================

servo_state = {

    "base": 90,
    "shoulder": 90,
    "elbow": 90,
    "wrist_pitch": 90,
    "wrist_roll": 90,
    "gripper": 0
}

# =========================================================
# INTERPOLATION
# =========================================================

def interpolate(current, target):

    return int(
        current +
        (target - current) *
        INTERPOLATION_SPEED
    )

# =========================================================
# GESTURE DETECTION
# =========================================================

def detect_gesture(landmarks):

    thumb = landmarks[4]
    index = landmarks[8]

    distance = math.sqrt(
        (thumb.x - index.x) ** 2 +
        (thumb.y - index.y) ** 2
    )

    if distance < 0.05:
        return "GRAB"

    return "OPEN"

# =========================================================
# SIMPLE IK
# =========================================================

def solve_ik(x, y, z, gesture):

    base = int(
        np.interp(x, [0,1], [0,180])
    )

    shoulder = int(
        np.interp(y, [0,1], [180,0])
    )

    elbow = int(
        np.interp(z, [-0.3,0.3], [40,140])
    )

    wrist_pitch = shoulder
    wrist_roll = base

    gripper = 180 if gesture == "GRAB" else 0

    return {

        "base": base,
        "shoulder": shoulder,
        "elbow": elbow,
        "wrist_pitch": wrist_pitch,
        "wrist_roll": wrist_roll,
        "gripper": gripper
    }

# =========================================================
# WEBSOCKET HANDLER
# =========================================================

async def websocket_handler(websocket):

    connected_clients.add(websocket)

    print("[INFO] WebSocket Client Connected")

    try:

        await websocket.wait_closed()

    finally:

        connected_clients.remove(websocket)

        print("[INFO] Client Disconnected")

# =========================================================
# WEBSOCKET TIMER
# =========================================================

async def websocket_timer():

    print("\n[INFO] Waiting for WebSocket client...\n")

    for i in range(WAIT_FOR_CLIENT_SECONDS, 0, -1):

        if connected_clients:

            print("[INFO] Client detected\n")
            return

        print(f"[WAIT] {i} seconds remaining...")

        await asyncio.sleep(1)

    print("\n[INFO] No client connected")
    print("[INFO] Running LOCAL prediction mode only\n")

# =========================================================
# MAIN TRACKING LOOP
# =========================================================

async def tracking_loop():

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print("[ERROR] Webcam not detected")
        return

    cv2.namedWindow(
        "Predictive Cloud Robotics",
        cv2.WINDOW_NORMAL
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        h, w, _ = frame.shape

        payload = None

        # =================================================
        # HAND TRACKING
        # =================================================

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                lm = hand_landmarks.landmark

                wrist = lm[0]

                # =========================================
                # FILTERS
                # =========================================

                x = kf_x.update(wrist.x)
                y = kf_y.update(wrist.y)
                z = kf_z.update(wrist.z)

                # =========================================
                # FUTURE PREDICTION
                # =========================================

                pred_x = kf_x.predict_future()
                pred_y = kf_y.predict_future()

                # =========================================
                # PIXEL POSITIONS
                # =========================================

                current_px = int(x * w)
                current_py = int(y * h)

                pred_px = int(pred_x * w)
                pred_py = int(pred_y * h)

                # =========================================
                # DRAW CURRENT
                # =========================================

                cv2.circle(
                    frame,
                    (current_px, current_py),
                    10,
                    (0,255,0),
                    -1
                )

                # =========================================
                # DRAW PREDICTION
                # =========================================

                cv2.circle(
                    frame,
                    (pred_px, pred_py),
                    15,
                    (0,0,255),
                    3
                )

                cv2.line(
                    frame,
                    (current_px, current_py),
                    (pred_px, pred_py),
                    (255,255,0),
                    2
                )

                # =========================================
                # GESTURE
                # =========================================

                gesture = detect_gesture(lm)

                # =========================================
                # IK
                # =========================================

                target_joints = solve_ik(
                    pred_x,
                    pred_y,
                    z,
                    gesture
                )

                # =========================================
                # INTERPOLATION
                # =========================================

                for joint in servo_state:

                    servo_state[joint] = interpolate(
                        servo_state[joint],
                        target_joints[joint]
                    )

                # =========================================
                # PAYLOAD
                # =========================================

                payload = {

                    "timestamp": time.time(),

                    "gesture": gesture,

                    "prediction": {

                        "x": pred_x,
                        "y": pred_y
                    },

                    "robot": servo_state
                }

        # =================================================
        # SEND ONLY IF CLIENT EXISTS
        # =================================================

        if payload and connected_clients:

            message = json.dumps(payload)

            disconnected = []

            for client in connected_clients:

                try:

                    await client.send(message)

                except:

                    disconnected.append(client)

            for dc in disconnected:

                connected_clients.remove(dc)

        # =================================================
        # DISPLAY WINDOW
        # =================================================

        cv2.imshow(
            "Predictive Cloud Robotics",
            frame
        )

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        await asyncio.sleep(0.001)

    cap.release()

    cv2.destroyAllWindows()

# =========================================================
# MAIN
# =========================================================

async def main():

    print(
        f"\n[INFO] WebSocket Server ws://{HOST}:{PORT}\n"
    )

    # START WEBSOCKET SERVER
    await websockets.serve(
        websocket_handler,
        HOST,
        PORT
    )

    # START TIMER IN BACKGROUND
    asyncio.create_task(
        websocket_timer()
    )

    # START TRACKING IMMEDIATELY
    await tracking_loop()

# =========================================================
# START
# =========================================================

asyncio.get_event_loop().run_until_complete(
    main()
)