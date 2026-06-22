import cv2
import mediapipe as mp
import pandas as pd
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

recording = False

dataset = []

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

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            wrist = hand_landmarks.landmark[0]

            x = wrist.x
            y = wrist.y
            z = wrist.z

            if recording:

                dataset.append([
                    time.time(),
                    x,
                    y,
                    z
                ])

            cv2.putText(
                frame,
                f"X:{x:.3f} Y:{y:.3f} Z:{z:.3f}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

    status = "RECORDING" if recording else "IDLE"

    cv2.putText(
        frame,
        status,
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    cv2.imshow(
        "Dataset Recorder",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord('s'):

        print("[INFO] Recording Started")
        recording = True

    elif key == ord('e'):

        print("[INFO] Recording Stopped")
        recording = False

    elif key == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

df = pd.DataFrame(
    dataset,
    columns=[
        "timestamp",
        "x",
        "y",
        "z"
    ]
)

df.to_csv(
    "gesture_wrist_dataset.csv",
    index=False
)

print(
    "[INFO] Saved:",
    len(df),
    "samples"
)