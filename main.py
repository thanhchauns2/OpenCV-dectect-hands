import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hlm in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hlm, mpHands.HAND_CONNECTIONS)

    pTime = time.time()
    fps = (pTime - cTime) / 1000
    # print(fps)
    cTime = pTime

    cv2.imshow('Image', img)
    cv2.waitKey(1)