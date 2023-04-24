import cv2
import numpy as np
import serial

ser = serial.Serial('COM3', 9600)

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

object_index = np.random.randint(0, 80)
object_name = classes[object_index]
print(f"Tracking {object_name}...")

suppression threshold
conf_threshold = 0.5
nms_threshold = 0.4

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
height, width, _ = frame.shape

pan_position = int(width / 2)
tilt_position = int(height / 2)

pan_step = 5
tilt_step = 5
pan_range = 180
tilt_range = 180

while True:
    ret, frame = cap.read()

    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == object_index:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

the closest one
    if len(indices) > 0:
        closest_index = 0
        closest_distance = np.inf
        for i in indices:
            i = i[0]
            x, y, w, h = boxes
