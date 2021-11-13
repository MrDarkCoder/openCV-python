import cv2
import numpy as np
import time

from modules import PoseEstimatiomModule as pm


cap = cv2.VideoCapture("Resources/AiTrainer/curls.mp4")


detector = pm.PoseDetector()

prevTime = 0
count = 0
directions = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    # img = cv2.imread("Resources/AiTrainer/test.jpg")

    img = detector.findPose(img, False)
    landmarks = detector.findPosition(img, draw=False)
    # print(landmarks)
    if len(landmarks) != 0:
        # Right Arm
        # detector.findAngle(img, 12, 14, 16, draw=True)
        # Left Arm
        angle = detector.findAngle(img, 11, 13, 15, draw=True)
        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (220, 310), (650, 100))
        # print(angle, per)
        # Check for the dumbbell curls
        # directions -> 0, 1 -> down, up as ONE count
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if directions == 0:
                count += 0.5
                directions = 1
        if per == 0:
            color = (0, 255, 0)
            if directions == 1:
                count += 0.5
                directions = 0

        # print(count)

        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)
        # Draw Curl Count
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

