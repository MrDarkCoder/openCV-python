import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)
# print(img)

''' coloring whole image '''
# img[:] = 256, 0, 0

''' coloring specific part [height, width] '''
# img[0:100, 100:200] = 256, 0, 0

''' Line (image, starting point, ending point, color, thickness) '''
cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 3)

''' Rectangle (image, starting point, ending point, diagonal points, color, thickness)'''
cv2.rectangle(img, (0, 0), (250, 250), (0, 0, 255), 3)

''' Circle (image, center point -> (starting point, ending point), radius, color, thickness) '''
cv2.circle(img, (400, 50), 30, (255, 255, 0), 5)

''' Text On Image (image, text, origin, font, scale, color, thickness)  '''
cv2.putText(img, " OPENCV  ", (300, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 3)

cv2.imshow("Image", img)

cv2.waitKey(0)
