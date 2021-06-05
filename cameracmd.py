import cv2
import numpy as np
import time

net = cv2.dnn.readNetFromDarknet(r"C:\Users\USER\Desktop\SIBI\python\abu\yolov4-tiny-custom.cfg", r"C:\Users\USER\Desktop\SIBI\python\abu\yolov4-tiny-custom_best.weights")
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
label = " "

cap = cv2.VideoCapture(0)
# timeframe = time.time()
# frame_id = 0

while 1:
    # _, frame = cap.read()
    # frame_id += 1
    
    ret, img = cap.read()
    img = cv2.resize(img,(680,480))
    hight,width,_ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)

    net.setInput(blob)

    output_layers_name = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_name)

    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.5,.4)

    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

        
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.8,.4)
    # Non-maximum suppression:
    results = [(class_ids[i], boxes[i]) for i in range(len(boxes)) if i in indexes]
        
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size =(len(boxes),3))
    if  len(indexes)>0:
        for i in indexes.flatten():
                     x,y,w,h = boxes[i]
                     if label != str(classes[class_ids[i]]) : 
                        label = str(classes[class_ids[i]])
                        print(label)
                     confidence = str(round(confidences[i],2))
                     color = colors[i]
                     cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                     # print(label)
                     # #if (label == 'A'):
                     #     cv2.putText(img,label + " " + confidence, (x,y+100),font,2,color,2)
                     #     print("A",confidence)
                     # if (label == 'B'):
                     #     cv2.putText(img,label + " " + confidence, (x,y+100),font,2,color,2)
                     #     print("B",confidence)
                     # if (label == 'C'):
                     #     cv2.putText(img,label + " " + confidence, (x,y+100),font,2,color,2)
                     #     print("C",confidence)
                     # if (label == 'D'):
                     #     cv2.putText(img,label + " " + confidence, (x,y+100),font,2,color,2)
                     #     print("D",confidence)
                     
    # elapsed_time = time.time() - timeframe
    # fps = frame_id / elapsed_time
    # cv2.putText(frame, str(round(fps,2)), (10,50), font, 2, (255, 255, 255), 2) #FPS value
    # cv2.putText(frame, "FPS", (220,50), font, 2, (255, 255, 255), 2) #FPS Label

    cv2.imshow('CAMERA',img)
    if cv2.waitKey(1) == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()