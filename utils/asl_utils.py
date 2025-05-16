import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_landmarks_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmark_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            if len(landmark_list) == 21:
                return np.array(landmark_list).flatten().reshape(1, 63, 1)
    return None
