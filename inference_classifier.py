import pickle

import cv2
import mediapipe as mp
import numpy as np

import time
from random import randint

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    "0": 'A', 
    "1": 'B', 
    "2": 'C', 
    "3": 'D', 
    "4": 'E', 
    "5": 'F', 
    "6": 'G', 
    "7": 'H', 
    "8": 'I', 
    "9": 'J', 
    "10": 'K', 
    "11": 'L', 
    "12": 'M', 
    "13": 'N', 
    "14": 'O', 
    "15": 'P', 
    "16": 'Q', 
    "17": 'R', 
    "18": 'S', 
    "19": 'T', 
    "20": 'U', 
    "21": 'V', 
    "22": 'W', 
    "23": 'X', 
    "24": 'Y', 
    "25": 'Z',
    "26": 'NEXT'
}

# for initial input_
inp_ = ''
words = ''

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        
        single_hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,  # image to draw
            single_hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        
        for i in range(len(single_hand_landmarks.landmark)):
            x = single_hand_landmarks.landmark[i].x
            y = single_hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        for i in range(len(single_hand_landmarks.landmark)):
            x = single_hand_landmarks.landmark[i].x
            y = single_hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[prediction[0]]
        
        # tracking the input
        temp_ = inp_
        inp_ = predicted_character

        if inp_ != temp_:
            print(inp_)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(90)


cap.release()
cv2.destroyAllWindows()

