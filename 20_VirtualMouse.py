import cv2
import numpy as np
import time

from modules import HandTrackingModule as htm

import autopy


wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 6

prevTime = 0
prevLocX, prevLocY = 0, 0
curLocX, curLocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
cap.set(10, 150)

detector = htm.HandDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)


while True:
    success, img = cap.read()

    # 01 find hand landmarks
    img = detector.findHands(img)
    landMarks, bbox = detector.findPosition(img)
    # print(landMarks)

    # 02 get Tip of Index and Middle finger
    if len(landMarks) != 0:
        x1, y1 = landMarks[8][1:]  # index finger
        x2, y2 = landMarks[12][1:]  # middle finger
        # print(x1, y1, x2, y2)

        # 03 Check Which Fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        # 04 Only Index Finger UP : MOVING MODE
        if fingers[1] == 1 and fingers[2] == 0:
            # 05 if moving mode: Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 06 Smoothen Values
            curLocX = prevLocX + (x3 - prevLocX) / smoothening
            curLocY = prevLocY + (y3 - prevLocY) / smoothening

            # 07 Move Mouse
            autopy.mouse.move(wScr - curLocX, curLocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            prevLocX, prevLocY = curLocX, curLocY

        # 08 Both Index and middle fingers are UP: CLICKING MODE
        if fingers[1] == 1 and fingers[2] == 1:
            # 09 if clicking mode: Find Distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                # 10 if clicking mode: Click mouse if Distance short
                autopy.mouse.click()

    # 11 Frame Rate
    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime

    # 12 Display
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
