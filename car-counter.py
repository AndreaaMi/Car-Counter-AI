import numpy as np
from ultralytics import YOLO
from sort import *
import cv2
import cvzone #to display detections
import math #to diplsay confident values on webcam screen

cap = cv2.VideoCapture("C:/Users/Andrea/Documents/GitHub/Car-Counter-AI/Videos/cars.mp4")  # za video

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask = cv2.imread("mask_for_cars_video.png")

#tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400,297,673,297]
totalCount = []

while True:
    succcess, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0,0))
    results = model(imgRegion, stream = True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1, y1, x2, y2 = box.xyxy[0] #easyer
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2-x1, y2-y1


            #confidence
            conf = math.ceil((box.conf[0]*100))/100


            #class name
            cls = int(box.cls[0])
            currentClasss = classNames[cls]

            if currentClasss == "car" or currentClasss == "truck" or currentClasss == "bus"\
                or currentClasss == "motorbike" and conf > 0.3:
                #cvzone.putTextRect(img, f'{currentClasss} {conf}', (max(0,x1), max(40,y1)), scale = 0.6, thickness = 1, offset = 3)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]),(limits[2], limits[3]), (0,0,255), 5) # red line

    for results in resultsTracker:
        x1,y1,x2,y2, Id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(results)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,255))
        cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(40, y1)), scale=2, thickness=3, offset=3)

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)

        if limits[0]<limits[2] and limits[1]-15<cy<limits[1]+15:
            if totalCount.count(Id) == 0: #if this id is not already present count it, if not 0
                totalCount.append(Id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)  # red line


    cv2.putText(img, str(len(totalCount)),(255,100), cv2.FONT_HERSHEY_PLAIN, 5, (50,50,255), 8)
    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
#turns camera on

