import cv2

print("Package Imported")

'''
reading image 
img = cv2.imread('Resources/lena.png')
cv2.imshow("outputImage", img)
cv2.waitKey(0)
'''
''' reading video 
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture("Resources/test_video.mp4")
while True:
    success, img = cap.read()
    img = cv2.resize(img, (frameWidth, frameHeight))
    cv2.imshow("Result", img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
'''
''' reading webcam 
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
# id 3 -> width
cap.set(3, frameWidth)
# id 4 -> height
cap.set(4, frameHeight)
# id 10 -> brightness
cap.set(10, 150)
while True:
    success, img = cap.read()
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
'''
