import cv2
import numpy as np

widthImg = 580
heightImg = 480

cap = cv2.VideoCapture(0)
# id 3 -> width
cap.set(3, widthImg)
# id 4 -> height
cap.set(4, heightImg)
# id 10 -> brightness
cap.set(10, 150)

''' 01 Preprocessing -> detect edges '''


def preprocessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDilation = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThre = cv2.erode(imgDilation, kernel, iterations=1)
    return imgThre


''' 02 Get Contours '''


def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    ''' when we add two points we can get the smallest for [0, 0]
     and biggest for [wImg, hImg] '''
    add = myPoints.sum(1)
    # print("add", add)
    # first
    myPointsNew[0] = myPoints[np.argmin(add)]  # smallest value index
    # last
    myPointsNew[3] = myPoints[np.argmax(add)]  # largest value index
    ''' when do diff we get negative point and positive point indicates 
     our diagonal points'''
    diff = np.diff(myPoints, axis=1)
    # left bottom
    myPointsNew[1] = myPoints[np.argmin(diff)]
    # right top
    myPointsNew[2] = myPoints[np.argmax(diff)]
    # print("NewPoints",myPointsNew)
    return myPointsNew


''' 03 Wrap Our Image '''


def getWarp(img, biggestContourPoints):
    biggestReorderedPoints = reorder(biggestContourPoints)
    pts1 = np.float32(biggestReorderedPoints)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    outputImage = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = outputImage[20:outputImage.shape[0] - 20, 20:outputImage.shape[1] - 20]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))

    return imgCropped


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


while True:
    success, img = cap.read()
    ''' Resizing '''
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()
    ''' Preprocessing '''
    imgThresold = preprocessing(img)


    biggestContourPoints = getContours(imgThresold)
    print(biggestContourPoints)
    if biggestContourPoints.size != 0:
        wrappedImg = getWarp(img, biggestContourPoints)
        imageArray = ([img,imgThresold],
                  [imgContour,wrappedImg])
        # imageArray = ([imgContour, wrappedImg])
        cv2.imshow("ImageWarped", wrappedImg)
    else:
        imageArray = ([img, imgThresold],
                      [img, img])
        # imageArray = ([imgContour, img])

    stackedImages = stackImages(0.6, imageArray)
    cv2.imshow("Webcam Result", stackedImages)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
