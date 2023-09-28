import cv2
import os

class Target:
    confidence = float(0.50) 
    nms = float(0.2)
    objects = []

class Model:
    data_directory = "data"
    classNames = []
    classFile = os.path.join('Object-Detection/',data_directory,'coco.names')
    with open(classFile,"rt") as f:
        classNames = f.read().rstrip("\n").split("\n")

    configPath = os.path.join('Object-Detection/',data_directory,"ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    weightsPath = os.path.join('Object-Detection/',data_directory,"frozen_inference_graph.pb")

    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

class Detection:
    def Objects(img, thres, nms, draw=True, objects=[]):
        classIds, confs, bbox = Model.net.detect(img,confThreshold=thres,nmsThreshold=nms)
        #print(classIds,bbox)
        if len(objects) == 0: objects = Model.classNames
        objectInfo =[]
        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                className = Model.classNames[classId - 1]
                if className in objects:
                    objectInfo.append([box,className])
                    if (draw):
                        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                        cv2.putText(img,Model.classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
                        cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)

        return img,objectInfo


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)


    while True:
        success, img = cap.read()
        result, objectInfo = Detection.Objects(img, thres=Target.confidence,nms=Target.nms, objects=Target.objects)
        #print(objectInfo)
        cv2.imshow("Live Feed",img)
        cv2.waitKey(1)
