import cv2
# import numpy as np
from time import sleep

from pynput.keyboard import Controller

from modules import HandTrackingModule as htm
from modules import utils


class Button:
    def __init__(self, position, text, size=[85, 85]):
        self.position = position
        self.text = text
        self.size = size

    def drawButton(self, img):
        x, y = self.position
        w, h = self.size
        cv2.rectangle(img, self.position, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, self.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
        return img

# instead of creating in class we are doing here to prevent this drawing repeatedly

#
# def drawAll(img, buttonList):
#     imgNew = np.zeros_like(img, np.uint8)
#     for button in buttonList:
#         x, y = button.pos
#         utils.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0], button.size[1]), 20, rt=0)
#         cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]), (255, 0, 255), cv2.FILLED)
#         cv2.putText(imgNew, button.text, (x + 40, y + 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
#
#     out = img.copy()
#     alpha = 0.5
#     mask = imgNew.astype(bool)
#     # print(mask.shape)
#     out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
#     return out


def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.position
        w, h = button.size
        utils.cornerRect(img, (button.position[0], button.position[1], button.size[0], button.size[1]), 20, rt=0)
        cv2.rectangle(img, button.position, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img


frameWidth = 1280
frameHeight = 720
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

detector = htm.HandDetector(detectionCon=0.8)
keyBoard = Controller()

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]


# drawing buttons
buttonList = []

for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 150, 100 * i + 100], key))

finalText = ""

while True:
    success, img = cap.read()

    # Find Hands
    img = detector.findHands(img)

    # landmarks
    landMarks, bbox = detector.findPosition(img)

    # Draw Button
    img = drawAll(img, buttonList)

    if len(landMarks) != 0:
        # print(landMarks[8])
        for button in buttonList:
            x, y = button.position
            w, h = button.size
            # print([x, y, w, h])
            if x < landMarks[8][1] < (x + w) & y < landMarks[8][2] < (y + h):
                # print("iM HERE")
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                length, _, _ = detector.findDistance(8, 12, img, draw=False)
                # print(length)

                # clicked
                if length < 15:
                    keyBoard.press(button.text)
                    cv2.rectangle(img, button.position, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    finalText += button.text
                    sleep(0.20)

    # cv2.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
    # cv2.putText(img, finalText, (60, 430), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
