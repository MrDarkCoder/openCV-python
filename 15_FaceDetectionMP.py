import cv2
import mediapipe as mp
import time

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

prevTime = 0
curTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(imgRGB)
    # print(results.detections)

    if results.detections:
        for idx, detect in enumerate(results.detections):
            # print(idx, detect)
            # print(detect.score)
            # print(detect.location_data.relative_bounding_box)  {xmin, ymin}
            # mpDraw.draw_detection(img, detect)
            bboxC = detect.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detect.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

''' 

0 
label_id: 0
score: 0.9589278697967529
location_data {
  format: RELATIVE_BOUNDING_BOX
  relative_bounding_box {
    xmin: 0.36165422201156616
    ymin: 0.44771724939346313
    width: 0.2266167402267456
    height: 0.30215591192245483
  }
  relative_keypoints {
    x: 0.4087764322757721
    y: 0.5224191546440125
  }
  relative_keypoints {
    x: 0.5095653533935547
    y: 0.5269309878349304
  }
  relative_keypoints {
    x: 0.4457020163536072
    y: 0.5885429382324219
  }
  relative_keypoints {
    x: 0.44878342747688293
    y: 0.6558127999305725
  }
  relative_keypoints {
    x: 0.36952120065689087
    y: 0.5562527179718018
  }
  relative_keypoints {
    x: 0.5855773687362671
    y: 0.5691821575164795
  }
}


'''