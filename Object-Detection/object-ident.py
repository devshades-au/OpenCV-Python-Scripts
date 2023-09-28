"""
Python script for object detection and labeling using OpenCV and COCO's dataset.

Version: 1.0
Author: Devshades
"""

import cv2
import os
import sys
sys.argv[0]

class Model:
    """
    Class representing the model data sets for object detection.

    Attributes:
        data_directory (str): Path where the model data is stored.
        classNames (list): List of class names for detected objects.
    """
    data_directory = "data"
    classNames = []

    # Read class names from a file
    classFile = os.path.join('Object-Detection/', data_directory, 'coco.names')
    with open(classFile, "rt") as f:
        classNames = f.read().rstrip("\n").split("\n")

    # Model paths
    configPath = os.path.join('Object-Detection/', data_directory, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    weightsPath = os.path.join('Object-Detection/', data_directory, "frozen_inference_graph.pb")

    # Initialize the model
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

class Detection:
    """
    Class for object detection in images.

    Methods:
        Objects(img, thres, nms, draw=True, objects=[]):
            Detect objects in an image and optionally draw bounding boxes and labels.

    """
    @staticmethod
    def Objects(img, thres, nms, draw=True, objects=[]):
        """
        Detect objects in an image and optionally draw bounding boxes and labels.

        Args:
            img (numpy.ndarray): Input image for object detection.
            thres (float): Confidence threshold for object detection.
            nms (float): Non-maximum suppression threshold to filter overlapping boxes.
            draw (bool): If True, draw bounding boxes and labels on the image.
            objects (list): List of specific objects to detect; if empty, detects all objects.

        Returns:
            img (numpy.ndarray): Image with bounding boxes and labels (if draw is True).
            objectInfo (list): List of detected objects and their bounding boxes.
        """
        classIds, confs, bbox = Model.net.detect(img, confThreshold=thres, nmsThreshold=nms)
        objectInfo = []

        if len(objects) == 0:
            objects = Model.classNames

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                className = Model.classNames[classId - 1]
                if className in objects:
                    objectInfo.append([box, className])
                    if draw:
                        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                        cv2.putText(img, Model.classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

        return img, objectInfo

if __name__ == "__main__":
    #Set the target configs
    thres = 0.50
    nms = 0.2
    objects = [] #eg. objects = ["person","cup","mobile phone"] empty = all objects
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        result, objectInfo = Detection.Objects(img, thres=thres, nms=nms, objects=objects)
        cv2.imshow("Live Feed", img)
        cv2.waitKey(1)
