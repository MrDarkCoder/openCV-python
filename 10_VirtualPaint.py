from typing import List, Union, Any

import cv2
import numpy as np

''' base color YELLOW, purple, green, blue '''
myColors = [[0, 29, 139, 255, 42, 255],
            [133, 56, 0, 159, 156, 255],
            [57, 76, 0, 100, 255, 255],
            [90, 48, 0, 118, 255, 255]]

myColorsRe = [[5, 107, 0, 19, 255, 255],
              [133, 56, 0, 159, 156, 255],
              [57, 76, 0, 100, 255, 255],
              [90, 48, 0, 118, 255, 255]]

myColorValues = [[51, 153, 255],  ## BGR
                 [255, 0, 255],
                 [0, 255, 0],
                 [255, 0, 0]]

myPoints = []  ## [x , y , colorId ]


def findColors(img, myColorsRe, myColorValues):
    """ Making HSV img """
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ''' Creating Mask Over those Values '''
    ''' Im GONNA use only YELLOW '''
    # lowerRange = np.array(myColors[0][0:3])
    # upperRange = np.array(myColors[0][3:6])
    # lowerRange = np.array([[hue_min, sat_min, val_min]])
    # upperRange = np.array([[hue_max, sat_max, val_max]])
    ''' For Detecting Every Color in the list '''
    count = 0
    newPoints = []
    for colors in myColorsRe:
        lowerRange = np.array(colors[0:3])
        upperRange = np.array(colors[3:6])
        maskImage = cv2.inRange(imgHSV, lowerRange, upperRange)
        # cv2.imshow(str(colors[0]), maskImage)
        x, y = getContours(maskImage)
        cv2.circle(imgResult, (x, y), 15, myColorValues[count], cv2.FILLED)
        if x != 0 and y != 0:
            newPoints.append([x, y, count])
        count += 1
        # cv2.imshow(str(color[0]),mask)
        return newPoints


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        ''' Draw in copied Img '''
        if area > 500:
            # cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    # sending the tip of the point
    return x + w // 2, y


def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)


frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

while True:
    success, img = cap.read()
    imgResult = img.copy()
    # findColors(img, myColorsRe)
    newPoints = findColors(img, myColorsRe, myColorValues)
    if len(newPoints) != 0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints) != 0:
        drawOnCanvas(myPoints, myColorValues)

    cv2.imshow("Webcam", imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
