import cv2
import numpy as np
import base64
import time
import pyrebase

#firebase
config = {
  "apiKey": "AIzaSyBlFG3O7ERh9L8PpcCn0LtD0gX88xTGh4w",
  "authDomain": "detectcar-b8366.firebaseapp.com",
  "databaseURL": "https://detectcar-b8366.firebaseio.com",
  "storageBucket": "detectcar-b8366.appspot.com"
}
firebase = pyrebase.initialize_app(config)
db = firebase.database()

# Load Yolo
net = cv2.dnn.readNetFromDarknet("E:/project/Final/ForTrain/CARYOLO.cfg","E:/project/Final/ForTrain/CARYOLO_final.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
classes = []
with open("E:/project/Final/ForTrain/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture('CAR1.mp4')

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

while True:
    _, frame = cap.read()
    frame_id += 1
    height, width, channels = frame.shape

    #draw
    mask = np.zeros(frame.shape, dtype=np.uint8)
    pts = np.array([[337,313],[0,630],[960,630],[736,313]], dtype=np.int32)
    # pts = np.array([[width/4,height/6],[0,height],[width,height],[(width/4)*3,height/6]], dtype=np.int32)
    #draw rectangle
    # pts = pts.reshape((-1,1,2))
    # draw = cv2.polylines(frame,[pts],True,(0,255,255))

    channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillConvexPoly(mask, pts, ignore_mask_color)

    # apply the mask
    masked_image = cv2.bitwise_and(frame, mask)

    # Detecting objects
    blob = cv2.dnn.blobFromImage(masked_image, 1/255, (412, 412), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    colors=[(0,0,255), (0,255,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0), (255,255,255)]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:

                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    currentframe = 0
    Dic_car = {"CAR":0 ,"VAN":0 ,"MOTORCYCLE":0 ,"BUS":0 ,"TRUCK":0}
    colors=[(255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,0,255)]
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            start=(x,y)
            end=(x + w, y + h)
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            #draw rectangle
            if classes[class_ids[i]] == "Motorcycle":
                cv2.rectangle(masked_image, start, end, colors[0], 1)
                cv2.putText(masked_image,"Motorcycle", (x, y + 30), font, 3, colors[0], 3)
            if classes[class_ids[i]] == "Car":
                cv2.rectangle(masked_image, start, end, colors[1], 1)
                cv2.putText(masked_image,"Car", (x, y + 30), font, 3, colors[1], 3)
            if classes[class_ids[i]] == "Van":
                cv2.rectangle(masked_image, start, end, colors[2], 1)
                cv2.putText(masked_image,"Van", (x, y + 30), font, 3, colors[2], 3)
            if classes[class_ids[i]] == "Bus":
                cv2.rectangle(masked_image, start, end, colors[3], 1)
                cv2.putText(masked_image,"Bus", (x, y + 30), font, 3, colors[3], 3)
            if classes[class_ids[i]] == "Truck":
                cv2.rectangle(masked_image, start, end, colors[4], 1)
                cv2.putText(masked_image,"Truck", (x, y + 30), font, 3, colors[4], 3)
            #count number
            if classes[class_ids[i]] == "Car":
                Dic_car["CAR"] += 1
            if classes[class_ids[i]] == "Van":
                Dic_car["VAN"] += 1
            if classes[class_ids[i]] == "Motorcycle":
                Dic_car["MORTORCYCLE"] += 1
            if classes[class_ids[i]] == "Bus":
                Dic_car["BUS"] += 1
            if classes[class_ids[i]] == "Truck":
                Dic_car["TRUCK"] += 1
    

# ###########image show#############
#     #draw
#     pts = np.array([[337,313],[0,630],[960,630],[736,313]], dtype=np.int32)
#     pts = pts.reshape((-1,1,2))
#     draw = cv2.polylines(frame,[pts],True,(0,255,255))

#     # Detecting objects
#     blob1 = cv2.dnn.blobFromImage(draw, 1/255, (412, 412), (0, 0, 0), True, crop=False)

#     net.setInput(blob1)
#     outs1 = net.forward(output_layers)

#     # Showing informations on the screen
#     class_ids1 = []
#     confidences1 = []
#     boxes1 = []
    
#     for out in outs1:

#         for detection in out:
#             scores1 = detection[5:]
#             class_id = np.argmax(scores1)
#             confidence = scores1[class_id]
#             if confidence > 0.2:

#                 # Object detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)

#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 boxes1.append([x, y, w, h])
#                 confidences1.append(float(confidence))
#                 class_ids1.append(class_id)

#     indexes1 = cv2.dnn.NMSBoxes(boxes1, confidences1, 0.8, 0.3)
#     currentframe1 = 0
#     for i in range(len(boxes1)):
#         if i in indexes1:
#             x, y, w, h = boxes1[i]
#             start=(x,y)
#             end=(x + w, y + h)
#             confidence = confidences1[i]
#             if classes[class_ids1[i]] == "Motorcycle":
#                 cv2.rectangle(draw, start, end, colors[0], 1)
#                 cv2.putText(draw,"Motorcycle", (x, y + 30), font, 3, colors[0], 3)
#             if classes[class_ids1[i]] == "Car":
#                 cv2.rectangle(draw, start, end, colors[1], 1)
#                 cv2.putText(draw,"Car", (x, y + 30), font, 3, colors[1], 3)
#             if classes[class_ids1[i]] == "Van":
#                 cv2.rectangle(draw, start, end, colors[2], 1)
#                 cv2.putText(draw,"Van", (x, y + 30), font, 3, colors[2], 3)
#             if classes[class_ids1[i]] == "Bus":
#                 cv2.rectangle(draw, start, end, colors[3], 1)
#                 cv2.putText(draw,"Bus", (x, y + 30), font, 3, colors[3], 3)
#             if classes[class_ids1[i]] == "Truck":
#                 cv2.rectangle(draw, start, end, colors[4], 1)
#                 cv2.putText(draw,"Truck", (x, y + 30), font, 3, colors[4], 3)
    #resize image
    scale_percent = 30 # percent of original size
    width = int(masked_image.shape[1] * scale_percent / 100)
    height = int(masked_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(masked_image, dim, interpolation = cv2.INTER_AREA)
    # base64
    cv2.imwrite("img_test.jpg",masked_image)
    with open("img_test.jpg", "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    # print(encoded_string)
    #firebase
    db.child("CAM/CAM1/").update({
    "Image":str(encoded_string),
    "CAR": Dic_car["CAR"],
    "VAN":Dic_car["VAN"],
    "MOTORCYCLE":Dic_car["MOTORCYCLE"],
    "BUS":Dic_car["BUS"],
    "TRUCK":Dic_car["TRUCK"]
    })

    db.child("CAM2/CAM2/").update({
    "Image":str(encoded_string),
    "CAR": Dic_car["CAR"],
    "VAN":Dic_car["VAN"],
    "MOTORCYCLE":Dic_car["MOTORCYCLE"],
    "BUS":Dic_car["BUS"],
    "TRUCK":Dic_car["TRUCK"]
    })
    # print("Car :: %s"%Dic_car["CAR"])
    # print("TRUCK :: %s"%Dic_car["TRUCK"])
    # print("VAN :: %s"%Dic_car["VAN"])
    # print("MOTORCYCLE :: %s"%Dic_car["MOTORCYCLE"])
    # print("BUS :: %s"%Dic_car["BUS"])

    cv2.imshow("Image", masked_image)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()