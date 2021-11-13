import cv2
import numpy as np

print("Package Imported")

img = cv2.imread('Resources/lambo.png')
''' (height = 462, width = 623, channels = 3) '''
print(img.shape)

''' args -> (width, height) '''
imgResize = cv2.resize(img, (300, 200))
print(imgResize.shape)

''' Cropping  '''
imgCropped = img[0:200, 200:500]

cv2.imshow("Lambo", img)
cv2.imshow("lambo resized", imgResize)
cv2.imshow("Cropped Image", imgCropped)

cv2.waitKey(0)
