import cv2
import mediapipe as mp
import time
import math


class PoseDetector:
    def __init__(self, mode=False, smoothness=True, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.smoothness = smoothness
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, smooth_landmarks=self.smoothness, min_detection_confidence=self.detectionConfidence, min_tracking_confidence=self.trackConfidence )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.myLandMarkPoints = []
        if self.results.pose_landmarks:
            for idx, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.myLandMarkPoints.append([idx, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.myLandMarkPoints

    # It will angle bet any three points
    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        # lm = [p1, x1, y1] -> that's why we are slicing [1: rest]
        x1, y1 = self.myLandMarkPoints[p1][1:]
        x2, y2 = self.myLandMarkPoints[p2][1:]
        x3, y3 = self.myLandMarkPoints[p3][1:]
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        # print(angle)
        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle


def main():
    frameWidth = 640
    frameHeight = 480
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, 150)

    prevTime = 0
    curTime = 0

    while True:
        success, img = cap.read()
        detector = PoseDetector()
        img = detector.findPose(img)

        landMarkPoints = detector.findPosition(img)

        if len(landMarkPoints) != 0:
            print(landMarkPoints[14])
            cv2.circle(img, (landMarkPoints[14][1], landMarkPoints[14][2]), 15, (0, 0, 255), cv2.FILLED)

        curTime = time.time()
        fps = 1 / (curTime - prevTime)
        prevTime = curTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)

        cv2.imshow("Webcam", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()


