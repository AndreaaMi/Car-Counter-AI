from ultralytics import YOLO
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
while True:
    succcess, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion, stream = True)
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
                cvzone.putTextRect(img, f'{currentClasss} {conf}', (max(0,x1), max(40,y1)), scale = 0.6, thickness = 1, offset = 3)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
#turns camera on

