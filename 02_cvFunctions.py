import cv2
import numpy as np

print("Package Imported")

''' Basic Functions '''
img = cv2.imread('Resources/lena.png')

''' Gray Scale  (image, color_space)'''
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

''' Blur (image, kernelSize(7,7), sigmaX)'''
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)

''' Edge Detector -> Canny '''
imgCanny = cv2.Canny(img, 150, 200)

''' Dilation -> to increase the thickness of edge '''
kernel = np.ones((5, 5), np.uint8)
imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)

''' Erosion -> to decrease thickness '''
imgErosion = cv2.erode(imgDilation, kernel, iterations=1)

# cv2.imshow("Gray Scale Image", imgGray)
# cv2.imshow("Gaussian Blur Image", imgBlur)
cv2.imshow("Canny Edge Detector Image", imgCanny)
cv2.imshow("Dilation Image", imgDilation)
cv2.imshow("Eroded Image", imgErosion)

cv2.waitKey(0)
