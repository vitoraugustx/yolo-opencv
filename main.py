# YOLO OpenCV object detection
import cv2 as cv
import numpy as np

net = cv.dnn.readNet("src/yolo_data/yolov3.weights", "src/yolo_data/yolov3.cfg")

classes = []
with open("src/yolo_data/YoloNames.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
file = "src/images/pedestrian_and_car.jpg"
img = cv.imread(file)
rate = 0.8
img = cv.resize(img, None, fx=rate, fy=rate)
height, width, channels = img.shape

# Detecting objects
blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv.FONT_HERSHEY_SIMPLEX   
fontScale = 0.5    
fontThickness = 2  
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        size, _ = cv.getTextSize(label, font, fontScale, fontThickness)
        color = colors[i]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.rectangle(img, (x - 1,y - 1), (x + size[0], y - 20), color, -1)
        cv.putText(img, label, (x, y - 5), font, fontScale, (0,0,0), fontThickness)

cv.imshow("Reconhecimento de objetos", img)
cv.waitKey(0)
cv.destroyAllWindows()
